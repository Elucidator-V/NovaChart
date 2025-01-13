import re
import json
import string
import os
from functools import cache
from typing import List, Dict, Union
from pydantic import BaseModel, Field
from loguru import logger
from pydantic_output_parser import PydanticOutputParser
from Levenshtein import ratio as levenshtein_ratio


class EvalUtils:
    symbols = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    @staticmethod
    def _clean(text: str) -> str:
        """Cleans text by removing symbols and converting to lowercase."""
        return ''.join(char.lower() for char in text if char not in EvalUtils.symbols)

    @staticmethod
    def similarity_string(str1: str, str2: str) -> float:
        return levenshtein_ratio(str1, str2)

    @staticmethod
    def similarity_number(num1: float, num2: float, delta: float = 1e-2) -> float:
        return max(0, (abs(num1) - abs(num1 - num2) + delta) / (abs(num1) + delta))

    @staticmethod
    @cache
    def _fuzzy_dist(seq1: List, seq2: List, i: int, j: int) -> int:
        if i == 0 and j == 0:
            return 0
        elif i == 0:
            return j
        elif j == 0:
            return i
        return min(
            EvalUtils._fuzzy_dist(seq1, seq2, i - 1, j) + 1,
            EvalUtils._fuzzy_dist(seq1, seq2, i, j - 1) + 1,
            EvalUtils._fuzzy_dist(seq1, seq2, i - 1, j - 1) + 1 - EvalUtils.similarity_object(seq1[i - 1], seq2[j - 1])
        )

    @staticmethod
    def similarity_sequence(seq1: List, seq2: List) -> float:
        EvalUtils._fuzzy_dist.cache_clear()
        return (len(seq1) + len(seq2) - EvalUtils._fuzzy_dist(seq1, seq2, len(seq1), len(seq2))) / (len(seq1) + len(seq2))

    @staticmethod
    def similarity_mapping(map1: Dict, map2: Dict) -> float:
        similarity = 0
        for k1, v1 in map1.items():
            max_key_similarity = -1
            selected_kv = None
            for k2, v2 in map2.items():
                key_similarity = EvalUtils.similarity_object(k1, k2)
                if key_similarity > max_key_similarity:
                    max_key_similarity = key_similarity
                    selected_kv = (k2, v2)
            similarity += max_key_similarity * EvalUtils.similarity_object(v1, selected_kv[1])
        return similarity / len(map1)

    @staticmethod
    def similarity_object(obj1: Union[int, float, str, List, Dict], obj2: Union[int, float, str, List, Dict]) -> float:
        if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
            return EvalUtils.similarity_number(obj1, obj2)
        elif isinstance(obj1, str) and isinstance(obj2, str):
            return EvalUtils.similarity_string(obj1, obj2)
        elif isinstance(obj1, list) and isinstance(obj2, list):
            return EvalUtils.similarity_sequence(obj1, obj2)
        elif isinstance(obj1, dict) and isinstance(obj2, dict):
            return EvalUtils.similarity_mapping(obj1, obj2)
        else:
            return 0

    @staticmethod
    def eval_exact(model: str, qkey: str, eval_gt_mapping: Dict, R: Dict, levenshtein: bool = False) -> Union[bool, float]:
        gt = eval_gt_mapping[qkey]['metadata']['answer']
        res = R[model][qkey]['text']
        if levenshtein:
            return levenshtein_ratio(EvalUtils._clean(gt), EvalUtils._clean(res))
        else:
            return EvalUtils._clean(gt) == EvalUtils._clean(res)

    @staticmethod
    def eval_extraction(model: str, qkey: str, eval_gt_mapping: Dict, R: Dict) -> float:
        gt_answer = eval_gt_mapping[qkey]['metadata']['answer'].replace("'", '"')
        res_answer = R[model][qkey]['text'].replace("'", '"')
        obj1 = json.loads(gt_answer)
        obj2 = json.loads(res_answer)
        return EvalUtils.similarity_object(obj1, obj2)

    @staticmethod
    def _extract_number(text: str) -> float:
        numbers = re.findall(r'\d+\.\d+|\d+', text)
        return float(numbers[-1]) if numbers else 0

    @staticmethod
    def eval_number(model: str, question_key: str, eval_gt_mapping: Dict, R: Dict) -> float:
        gt_number = float(eval_gt_mapping[question_key]['metadata']['answer'])
        try:
            res_number = EvalUtils._extract_number(R[model][question_key]['text'])
        except Exception as e:
            logger.warning(f"Failed to extract number: {e}")
            return 0
        else:
            return EvalUtils.similarity_number(gt_number, res_number)


chart_type_mapping = {
    'singlelineplot': "single-class line plot",
    'multilineplot': "multi-class line plot", 
    'multilineplot_addition': "multi-class line plot",
    'singlescatterplot': "single-class scatter plot",
    'multiscatterplot': "multi-class scatter plot",
    'univariatehistogram': "univariate histogram",
    'bivariatehistogram': "bivariate histogram",
    'singlecountplot': "single-hue bar plot",
    'multicountplot': "multi-hue bar plot",
    'boxplot': "box plot", 
    'heatmap': "heatmap", 
    'knowledgegraph': "knowledge graph",
    'piechart': "pie chart",
    'ringchart': "ring chart",
    'rosechart': "rose chart",
    'radarchart': "radar chart", 
    'sankeychart': "sankey chart",
    'wordcloud': "word cloud",
    'table': "table"
}

criteria = {
    "analysis": """1. **Correctness:**
   - 0: Output contains significant factual errors or misinterpretations of data.
   - 1: Output contains several factual errors or misinterpretations of data.
   - 2: Output contains some factual errors or misinterpretations of data.
   - 3: Output is mostly accurate but may contain minor errors or ambiguities.
   - 4: Output is largely accurate with few errors or ambiguities.
   - 5: Output is completely accurate with no factual errors or ambiguities.
2. **Meaningfulness:**
   - 0: Output lacks meaningful insights or fails to provide relevant analysis.
   - 1: Output provides very limited or superficial insights.
   - 2: Output provides some insights but lacks depth or relevance.
   - 3: Output provides relevant insights with reasonable depth.
   - 4: Output provides insightful analysis with good depth and relevance.
   - 5: Output provides highly insightful analysis with deep relevance and clarity.""",
    "code": """1. **Correctness:**
   - 0: Code does not meet any of the requirements of the task and produces completely incorrect or irrelevant or endless repetition results.
   - 1: Code contains major errors or misinterpretations or attempt to evade most of task requirements (e.g. let users load data themselves).
   - 2: Code meets some requirements of the task but contains notable errors.
   - 3: Code mostly meets the requirements of the task with minor errors or a general rather than detailed code.
   - 4: Code largely meets the requirements of the task with few errors or ambiguities.
   - 5: Code perfectly meets all requirements of the task.
2. **Quality:**
   - 0: Code lacks any discernible structure or is definitely not directly executable.
   - 1: Code has minimal structure and inconsistent formatting, making it difficult to follow.
   - 2: Code has basic structure but lacks consistency in formatting and naming conventions.
   - 3: Code has a clear structure and follows consistent formatting and naming conventions, but improvements can be made for better readability.
   - 4: Code has a well-defined structure, consistent formatting, and clear variable naming, contributing to good readability.
   - 5: Code has an excellent structure, impeccable formatting, and meaningful variable naming, making it highly readable and maintainable."""
}


class AnalysisResponse(BaseModel):
    Model_A_Evaluation: str = Field(description="A brief comment on model A's output. ")
    Model_A_Correctness: int = Field(description="The correctness score (0~5) of model A's output. ")
    Model_A_Meaningfulness: int = Field(description="The meaningfulness score (0~5) of model A's output. ")
    Model_B_Evaluation: str = Field(description="A brief comment on model B's output. ")
    Model_B_Correctness: int = Field(description="The correctness score (0~5) of model B's output. ")
    Model_B_Meaningfulness: int = Field(description="The meaningfulness score (0~5) of model B's output. ")
    winner: str = Field(description="Which model's output is better, A or B? ")


class CodeResponse(BaseModel):
    Model_A_Evaluation: str = Field(description="A brief comment on model A's output. ")
    Model_A_Correctness: int = Field(description="The correctness score (0~5) of model A's output. ")
    Model_A_Quality: int = Field(description="The quality score (0~5) of model A's output. ")
    Model_B_Evaluation: str = Field(description="A brief comment on model B's output. ")
    Model_B_Correctness: int = Field(description="The correctness score (0~5) of model B's output. ")
    Model_B_Quality: int = Field(description="The quality score (0~5) of model B's output. ")


analysis_parser = PydanticOutputParser(pydantic_object=AnalysisResponse)
code_parser = PydanticOutputParser(pydantic_object=CodeResponse)


def trim_output(output: str) -> str:
    if len(output) > 3000:
        return output[:3000] + f"(...{len(output) - 3000} more chars here, you should check whether the output contains meaningless endless repetition. )"
    return output


class PromptGenerator:
    def __init__(self, subtable_root: str, eval_analysis_template: string.Template, eval_code_template: string.Template):
        self.subtable_root = subtable_root
        self.eval_analysis_template = eval_analysis_template
        self.eval_code_template = eval_code_template

    def make_prompt(self, chart_type: str, sub_key: str, task_type: str, task: str, model_a_output: str, model_b_output: str) -> str:
        json_fpath = os.path.join(self.subtable_root, chart_type, sub_key + ".json")
        with open(json_fpath, 'r') as fp:
            json_obj = json.load(fp)
        template = self.eval_analysis_template if task_type == "analysis" else self.eval_code_template
        return template.substitute(
            chart_type=chart_type_mapping[chart_type],
            description=json_obj['lf_description'],
            semantic=json_obj['semantic'],
            lf_data=json_obj['data_lf'],
            task=task,
            model_a_output=trim_output(model_a_output),
            model_b_output=trim_output(model_b_output),
            criteria=criteria[task_type],
            response_format=analysis_parser.get_format_instructions() if task_type == "analysis" else code_parser.get_format_instructions()
        )
