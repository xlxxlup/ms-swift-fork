# Copyright (c) ModelScope Contributors. All rights reserved.
# Outcome Reward Model (ORM) implementations for GRPO training.

import json
import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from swift.infer_engine import InferRequest
from swift.utils import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from swift.megatron.arguments import MegatronArguments
    from swift.rlhf_trainers import GRPOConfig


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous outcome reward models (ORM).

    Use this for reward functions that involve I/O operations (e.g., API calls,
    database queries) that can benefit from async execution.

    Async reward functions are executed in parallel using asyncio.gather,
    which can significantly speed up reward computation when multiple async
    reward functions are used or when the reward function involves network calls.

    Example:
        class MyAsyncReward(AsyncORM):
            async def __call__(self, completions, **kwargs) -> List[float]:
                # Use asyncio.gather for parallel execution of all API calls
                import asyncio
                import aiohttp

                async def score_single(session, text):
                    async with session.post(api_url, json={'text': text}) as resp:
                        result = await resp.json()
                        return result['score']

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, c) for c in completions]
                    rewards = await asyncio.gather(*tasks)
                    return list(rewards)
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class MathAccuracy(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, accuracy_orm=None):
        super().__init__(args)
        self.min_len_value_wrong = args.cosine_min_len_value_wrong
        self.max_len_value_wrong = args.cosine_max_len_value_wrong
        self.min_len_value_correct = args.cosine_min_len_value_correct
        self.max_len_value_correct = args.cosine_max_len_value_correct
        self.max_len = args.cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        self.ngram_size = args.repetition_n_grams
        self.max_penalty = args.repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        assert args.soft_cache_length < args.soft_max_length
        self.soft_max_length = args.soft_max_length
        self.soft_cache_length = args.soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}

# =============================================================================
# Custom Dermatology Reward Functions
# =============================================================================

def _extract_dermatology_scores(response: str) -> str:
    """从模型输出中提取分数"""
    if not response or not isinstance(response, str):
        return None
    last_score_pos = response.rfind('<score>')
    if last_score_pos == -1:
        return None
    after_last_score = response[last_score_pos:]
    score_match = re.search(r'<score>\s*最终得分[：:]\s*(.+?)\s*</score>', after_last_score, re.DOTALL)
    if not score_match:
        return None
    score_text = score_match.group(1).strip()
    if '候选' in score_text:
        candidate2_pos = score_text.find('候选2:')
        if candidate2_pos > 0:
            candidate1_part = score_text[score_text.find('候选1:') + len('候选1:'):candidate2_pos].strip(' ,')
            candidate2_part = score_text[candidate2_pos + len('候选2:'):].strip(' ,')
            candidate1_part = re.sub(r'=.*$', '', candidate1_part).strip()
            candidate2_part = re.sub(r'=.*$', '', candidate2_part).strip()
            score1_full = re.match(r'^(\d+\.?\d*(?:\[[^\]]*\])?)', candidate1_part)
            score2_full = re.match(r'^(\d+\.?\d*(?:\[[^\]]*\])?)', candidate2_part)
            s1 = score1_full.group(1) if score1_full else candidate1_part
            s2 = score2_full.group(1) if score2_full else candidate2_part
            return f'{s1},{s2}'
    return score_text


def _parse_main_scores(scores_text: str) -> tuple:
    """从分数字符串中提取两个主分数"""
    if not scores_text:
        return None, None
    parts = []
    current = []
    bracket_level = 0
    for char in str(scores_text):
        if char == '[':
            bracket_level += 1
            current.append(char)
        elif char == ']':
            bracket_level -= 1
            current.append(char)
        elif char == ',' and bracket_level == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current))
    if len(parts) != 2:
        return None, None
    s1_match = re.match(r'^(\d+\.?\d*)', parts[0].strip())
    s2_match = re.match(r'^(\d+\.?\d*)', parts[1].strip())
    if s1_match and s2_match:
        return float(s1_match.group(1)), float(s2_match.group(1))
    return None, None


def _split_dermatology_scores(scores_text: str) -> tuple:
    """将分数字符串分割为两部分"""
    if not scores_text:
        return None, None
    parts = []
    current = []
    bracket_level = 0
    for char in str(scores_text):
        if char == '[':
            bracket_level += 1
            current.append(char)
        elif char == ']':
            bracket_level -= 1
            current.append(char)
        elif char == ',' and bracket_level == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current))
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, None


def _extract_rule_ids(score_part: str) -> List[str]:
    """从分数部分提取规则编号"""
    if not score_part:
        return []
    bracket_match = re.search(r'\[([^\]]+)\]', score_part)
    if not bracket_match:
        return []
    content = bracket_match.group(1)
    rules = []
    for item in content.split(','):
        item = item.strip()
        if item and not item.startswith('best') and not item.startswith('+'):
            rule_match = re.match(r'^([\d.]+)', item)
            if rule_match:
                rules.append(rule_match.group(1))
    return rules


class CorrectnessReward(ORM):
    """正确性奖励：模型输出的分数相对关系与 reference_scores 一致 → +2分"""
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

    def __call__(self, completions: List[str], reference_scores: str = None, **kwargs) -> List[float]:
        # 获取 reference_scores 列表（每个 completion 对应一个 reference_scores）
        ref_scores_list = None

        if reference_scores is not None:
            # 如果是单个字符串，转换为列表（所有 completion 共用）
            if isinstance(reference_scores, str):
                ref_scores_list = [reference_scores] * len(completions)
            elif isinstance(reference_scores, list):
                ref_scores_list = reference_scores
        else:
            # 从 kwargs 中获取
            ref_scores_list = kwargs.get('reference_scores')
            if ref_scores_list is None:
                ref_scores_list = kwargs.get('reference_scores_list', [])

        # 如果还是单个字符串，转换为列表
        if isinstance(ref_scores_list, str):
            ref_scores_list = [ref_scores_list] * len(completions)

        if not ref_scores_list or len(ref_scores_list) == 0:
            logger.warning(f"CorrectnessReward: reference_scores is empty. kwargs keys: {list(kwargs.keys())}")
            return [-2.0] * len(completions)

        rewards = []
        for i, completion in enumerate(completions):
            # 获取当前 completion 对应的 reference_scores
            if i < len(ref_scores_list):
                ref_scores = ref_scores_list[i]
            else:
                ref_scores = ref_scores_list[0]

            # 解析参考分数的相对关系
            ref_s1, ref_s2 = _parse_main_scores(ref_scores)
            if ref_s1 is None or ref_s2 is None:
                rewards.append(-2.0)
                continue

            # 参考关系：True表示候选1更好，False表示候选2更好
            ref_relation = ref_s1 > ref_s2

            # 提取模型输出的分数
            scores_text = _extract_dermatology_scores(completion)
            if not scores_text:
                rewards.append(-2.0)
                continue

            s1, s2 = _parse_main_scores(scores_text)
            if s1 is None or s2 is None:
                rewards.append(-2.0)
                continue

            # 检查模型输出的相对关系是否与参考一致
            model_relation = s1 > s2

            if model_relation == ref_relation:
                rewards.append(2.0)  # 相对关系一致
            else:
                rewards.append(-2.0)  # 相对关系不一致

        return rewards


class RuleMatchReward(ORM):
    """规则匹配奖励：每匹配一个规则 → +1分"""
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

    def __call__(self, completions: List[str], reference_scores: str = None, **kwargs) -> List[float]:
        # 获取 reference_scores 列表（每个 completion 对应一个 reference_scores）
        ref_scores_list = None

        if reference_scores is not None:
            # 如果是单个字符串，转换为列表（所有 completion 共用）
            if isinstance(reference_scores, str):
                ref_scores_list = [reference_scores] * len(completions)
            elif isinstance(reference_scores, list):
                ref_scores_list = reference_scores
        else:
            # 从 kwargs 中获取
            ref_scores_list = kwargs.get('reference_scores')
            if ref_scores_list is None:
                ref_scores_list = kwargs.get('reference_scores_list', [])

        # 如果还是单个字符串，转换为列表
        if isinstance(ref_scores_list, str):
            ref_scores_list = [ref_scores_list] * len(completions)

        if not ref_scores_list or len(ref_scores_list) == 0:
            # 调试：打印 kwargs 的所有键
            logger.warning(f"RuleMatchReward: reference_scores is empty. kwargs keys: {list(kwargs.keys())}")
            return [0.0] * len(completions)

        rewards = []
        for i, completion in enumerate(completions):
            # 获取当前 completion 对应的 reference_scores
            if i < len(ref_scores_list):
                ref_scores = ref_scores_list[i]
            else:
                ref_scores = ref_scores_list[0]

            # 解析参考分数，提取参考规则
            ref_score1_part, ref_score2_part = _split_dermatology_scores(ref_scores)
            if not ref_score1_part or not ref_score2_part:
                rewards.append(0.0)
                continue

            ref_rules1 = set(_extract_rule_ids(ref_score1_part))
            ref_rules2 = set(_extract_rule_ids(ref_score2_part))

            # 从模型输出提取分数
            scores_text = _extract_dermatology_scores(completion)
            if not scores_text:
                rewards.append(0.0)
                continue

            # 解析模型输出的规则
            model_score1_part, model_score2_part = _split_dermatology_scores(scores_text)
            if not model_score1_part or not model_score2_part:
                rewards.append(0.0)
                continue

            model_rules1 = set(_extract_rule_ids(model_score1_part))
            model_rules2 = set(_extract_rule_ids(model_score2_part))

            # 按位置匹配（防止投机取巧）
            # 候选1的模型规则 vs 候选1的参考规则
            # 候选2的模型规则 vs 候选2的参考规则
            match1 = len(model_rules1 & ref_rules1)
            match2 = len(model_rules2 & ref_rules2)
            match_count = match1 + match2

            # 计算精度惩罚（防止模型输出过多无关规则）
            # 精度 = 命中规则数 / 输出规则总数
            precision1 = match1 / len(model_rules1) if model_rules1 else 0.0
            precision2 = match2 / len(model_rules2) if model_rules2 else 0.0
            avg_precision = (precision1 + precision2) / 2

            # 最终奖励 = 匹配数 × 精度
            # 这样正确输出能得高分，投机输出（堆砌规则）会因为精度低而被惩罚
            final_reward = float(match_count * avg_precision)*2

            rewards.append(final_reward)

        return rewards


class OrderConsistencyReward(ORM):
    """顺序一致性奖励：每2条数据为一组，得分相对大小关系一致 → 各+2分"""
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        num_completions = len(completions)
        scores_data = []
        for completion in completions:
            scores_text = _extract_dermatology_scores(completion)
            s1, s2 = _parse_main_scores(scores_text) if scores_text else (None, None)
            scores_data.append((s1, s2))
        for i, (s1, s2) in enumerate(scores_data):
            if s1 is None or s2 is None:
                rewards.append(0.0)
            else:
                rewards.append(0.0)
        for i in range(0, num_completions, 2):
            if i + 1 >= num_completions:
                break
            s1_a, s2_a = scores_data[i]
            s1_b, s2_b = scores_data[i + 1]
            if s1_a is not None and s2_a is not None and s1_b is not None and s2_b is not None:
                if (s1_a < s2_a) and (s1_b > s2_b):
                    rewards[i] = 2.0
                    rewards[i + 1] = 2.0
                elif (s1_a > s2_a) and (s1_b < s2_b):
                    rewards[i] = 2.0
                    rewards[i + 1] = 2.0
        return rewards


# Register custom reward functions
orms['correctness'] = CorrectnessReward
orms['rule_match'] = RuleMatchReward
orms['order_consistency'] = OrderConsistencyReward
logger.info('Registered custom dermatology reward functions: correctness, rule_match, order_consistency')
