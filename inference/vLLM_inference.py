print("=== Drug Drug Interaction Reasoning Generate Script (vLLM Version) ===")
print("Importing libraries...")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import re
import os
import json
from prompts import system_prompt_v23_role as system_prompt

# VLLM V1 only: Turn off multiprocessing to make the scheduling deterministic.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

print("All libraries imported successfully!")

class DrugInteractionPredictor:
    def __init__(self, model_path, model_type, lora_path, save_thinking, backend: str | None = None):
        """初始化模型（vLLM 或 OpenAI 后端）"""
        print("Initializing vLLM model...")

        self.backend = (backend or DEFAULT_BACKEND).lower()
        self.model_type = model_type
        self.model_path = model_path
        self.lora_path = lora_path
        self.lora_name = "ddi_lora_adapter"
        self.save_thinking = save_thinking
        self.system_prompt = system_prompt

        # vLLM backend init (optional)
        self.llm = None
        self.sampling_params = None

        if self.backend == "openai":
            if self.lora_path:
                print("Warning: LoRA is ignored on OpenAI backend (vLLM-only).")
            # Keep these aligned with existing defaults for parity
            self.sampling_params = SamplingParams(
                temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=2048, stop=None
            )
            self.tokenizer = None
            return

        llm_args = {
            "model": self.model_path,
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 16384,
            "enforce_eager": True,
        }
        if self.lora_path:
            print(f"Enabling LoRA support. Adapter path: {self.lora_path}")
            llm_args["enable_lora"] = True
            llm_args["max_loras"] = 1
            llm_args["max_lora_rank"] = 8

        self.llm = LLM(**llm_args)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            print("Tokenizer loaded successfully for chat template")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            max_tokens=2048,
            stop=None
        )
        
        self.save_thinking = save_thinking

        # 添加训练数据存储变量
        self.ddi_db = None
        self.not_find_count = 0
        self.sim_rank_df = None

        # 添加知识图谱信息存储变量
        self.kg_path = None
        # 添加未找到键的记录列表
        self.not_found_keys = []
        
    def load_data(self, input_file, drugs_file, kg_path=None):
        self.input_file = pd.read_csv(input_file)
        self.drugs_df = pd.read_csv(drugs_file)

       
        
        if kg_path and os.path.exists(kg_path):
            self.kg_path = {}
            with open(kg_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        kg_data = json.loads(line.strip())
                        # 使用药物ID对作为键
                        key = (kg_data['drug1_db'], kg_data['drug2_db'])
                        self.kg_path[key] = kg_data['path_str']
                    except json.JSONDecodeError:
                        continue
            print(f"Loaded knowledge graph paths for {len(self.kg_path)} drug pairs")

        # 创建药物ID到信息的映射
        self.drug_info_map = {}
        for _, row in self.drugs_df.iterrows():
            self.drug_info_map[row['id']] = {
                'name': row['name'],
                'description': row['description'],
                'indication': row['indication'],
                'pharmacodynamics': row['pharmacodynamics'],
                'moa': row['moa'],
                'absorption': row['absorption'],
                'distribution': row['distribution'],
                'metabolism': row['metabolism'],
                'excretion': row['excretion'],
                'toxicity': row['toxicity'],
                'smiles': row['smiles']
            }

    def get_kg_path_info(self, drug1_id, drug2_id, row_index=None):
        """获取药物对的知识图谱路径信息"""
        if self.kg_path is None:
            return None
        
        key1 = (drug1_id, drug2_id)
        # key2 = (drug2_id, drug1_id)
        
        # path_str = self.kg_path.get(key1) or self.kg_path.get(key2)
        path_str = self.kg_path.get(key1)

        if path_str:
            # print(f"Found knowledge graph path for {key1}")  # 添加调试信息

            kg_path_list = [
            f"## Knowledge Graph Information",
            path_str,
            ]

            # 使用换行符连接所有部分
            kg_path = "\n".join(kg_path_list)
            
            return kg_path.strip()

        else:
            self.not_find_count += 1
            # 记录未找到的键信息
            not_found_info = {
                'row_index': row_index,
                'drug1_id': drug1_id,
                'drug2_id': drug2_id,
                'key1': key1,
                # 'key2': key2,
                'drug1_name': self.drug_info_map.get(drug1_id, {}).get('name', f'Unknown_{drug1_id}'),
                'drug2_name': self.drug_info_map.get(drug2_id, {}).get('name', f'Unknown_{drug2_id}')
            }
            self.not_found_keys.append(not_found_info)

            print(f"Total not found count: {self.not_find_count}")  # 添加调试信息

        return None
    
    def save_not_found_keys_analysis(self, output_file):
        if not self.not_found_keys:
            print("No missing keys to save.")
            return
        
        # 创建DataFrame用于保存详细信息
        not_found_df = pd.DataFrame(self.not_found_keys)
        
        # 保存详细的未找到键信息
        not_found_output = output_file.replace('.csv', '_kg_not_found_keys.csv')
        not_found_df.to_csv(not_found_output, index=False)
        print(f"Saved {len(self.not_found_keys)} missing KG keys to {not_found_output}")
        
        # 创建统计分析文件
        analysis_output = output_file.replace('.csv', '_kg_missing_analysis.txt')
        with open(analysis_output, 'w', encoding='utf-8') as f:
            f.write("=== Knowledge Graph Missing Keys Analysis ===\n\n")
            f.write(f"Total missing keys: {len(self.not_found_keys)}\n")
            f.write(f"Total KG keys available: {len(self.kg_path) if self.kg_path else 0}\n\n")
            
            if self.kg_path:
                # 分析可用键的格式
                sample_available_keys = list(self.kg_path.keys())[:10]
                f.write("Sample available keys (first 10):\n")
                for i, key in enumerate(sample_available_keys, 1):
                    f.write(f"  {i}. {key} (types: {type(key[0])}, {type(key[1])})\n")
                f.write("\n")
            
            # 分析缺失键的格式
            f.write("Sample missing keys (first 10):\n")
            for i, item in enumerate(self.not_found_keys[:10], 1):
                f.write(f"  {i}. key1: {item['key1']} (types: {type(item['drug1_id'])}, {type(item['drug2_id'])})\n")
                # f.write(f"      key2: {item['key2']}\n")
                # f.write(f"      drugs: {item['drug1_name']} - {item['drug2_name']}\n")
            f.write("\n")
            
            # 统计不同类型的药物ID
            if self.not_found_keys:
                drug1_ids = [item['drug1_id'] for item in self.not_found_keys]
                drug2_ids = [item['drug2_id'] for item in self.not_found_keys]
                all_missing_ids = drug1_ids + drug2_ids
                
                f.write(f"Unique missing drug1_ids: {len(set(drug1_ids))}\n")
                f.write(f"Unique missing drug2_ids: {len(set(drug2_ids))}\n")
                f.write(f"Total unique missing drug IDs: {len(set(all_missing_ids))}\n\n")
                
                # 检查ID类型
                if all_missing_ids:
                    id_types = {}
                    for drug_id in set(all_missing_ids):
                        id_type = type(drug_id).__name__
                        id_types[id_type] = id_types.get(id_type, 0) + 1
                    
                    f.write("Missing drug ID types distribution:\n")
                    for id_type, count in id_types.items():
                        f.write(f"  {id_type}: {count}\n")
            
            f.write(f"\nAnalysis saved at: {analysis_output}\n")
        
        print(f"Missing keys analysis saved to {analysis_output}")
    def create_user_prompt(self, drug1_id, drug2_id, row_index=None):
        """为单个药物对创建用户提示词"""
        drug1_info = self.drug_info_map.get(drug1_id, {})
        drug2_info = self.drug_info_map.get(drug2_id, {})
        
        # 处理空值，只保留有内容的属性
        def format_drug_info(drug_info, generic_drug_name):
            actual_drug_name = drug_info.get('name', generic_drug_name)
            info_lines = []
            
            # 定义属性映射
            attributes = {
                'name': 'name',
                'description': 'description',
                'indication': 'indication',
                'pharmacodynamics': 'pharmacodynamics',
                'moa': 'mechanism of action',
                'absorption': 'absorption',
                'distribution': 'distribution',
                'metabolism': 'metabolism',
                'excretion': 'excretion',
                'toxicity': 'toxicity'
            }
            
            for key, display_name in attributes.items():
                value = drug_info.get(key, '')
                # 检查值是否有效（不为空、不为NaN、不为None）
                if value and pd.notna(value) and str(value).strip() != '' and str(value).lower() != 'nan':
                    if key == 'name':
                        prefix = generic_drug_name
                    else:
                        prefix = actual_drug_name
                    info_lines.append(f"{prefix} {display_name}: {str(value).strip()}")
            
                    # info_lines.append(f"{drug_name} {display_name}: {str(value).strip()}")
            
            return "\n".join(info_lines)
        
        # 获取药物名称
        drug1_name = drug1_info.get('name', 'Drug1')
        drug2_name = drug2_info.get('name', 'Drug2')
        
        # 格式化药物信息
        drug1_formatted_info = format_drug_info(drug1_info, 'Drug1')
        drug2_formatted_info = format_drug_info(drug2_info, 'Drug2')
        
        # 构建完整的提示词
        prompt_parts = [
            f"Determine the interaction type or therapeutic indication when {drug1_name} and {drug2_name} are used together.",
            f"## Drug1 Information \n{drug1_formatted_info}",
            f"## Drug2 Information \n{drug2_formatted_info}"
        ]

        # 添加知识图谱路径信息
        kg_path_info = self.get_kg_path_info(drug1_id, drug2_id, row_index=row_index)
        if kg_path_info:
            prompt_parts.append(kg_path_info)

        # 使用换行符连接所有部分
        user_prompt = "\n\n".join(prompt_parts)
       
        return user_prompt.strip()
    
    def format_chat_prompt(self, user_prompt, model_type="qwen"):
        """格式化聊天提示词（自动使用模型的chat_template）"""
        try:
            # 如果还没有加载tokenizer，则加载
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    enable_thinking=False
                )
            
            # 构建消息格式
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 使用tokenizer的apply_chat_template方法
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                return formatted_prompt
            else:
                # 如果没有chat_template，回退到手动格式化
                print(f"Warning: No chat_template found for model, using manual formatting for {model_type}")
                return self._manual_format_chat_prompt(user_prompt, model_type)
                
        except Exception as e:
            print(f"Error using chat_template: {e}, falling back to manual formatting")
            return self._manual_format_chat_prompt(user_prompt, model_type)
    
    def _manual_format_chat_prompt(self, user_prompt, model_type="qwen"):
        """手动格式化聊天提示词（作为备用方案）"""
        if model_type.lower() == "qwen":
            # Qwen的聊天格式
            formatted_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif model_type.lower() == "gemma" or "medgemma" in model_type.lower():
            # Gemma/MedGemma的聊天格式
            formatted_prompt = f"<start_of_turn>user\n{self.system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif model_type.lower() == "llama":
            # Llama的聊天格式
            formatted_prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            # 默认格式（简单拼接）
            formatted_prompt = f"System: {self.system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        return formatted_prompt

    def format_chat_prompt_backup(self, user_prompt, model_type="qwen"):
        """格式化聊天提示词（支持多种模型格式）"""
        if model_type.lower() == "qwen":
            # Qwen的聊天格式
            formatted_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif model_type.lower() == "gemma" or "medgemma" in model_type.lower():
            # Gemma/MedGemma的聊天格式
            formatted_prompt = f"<start_of_turn>user\n{self.system_prompt}\n\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif model_type.lower() == "llama":
            # Llama的聊天格式
            formatted_prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        else:
            # 默认格式（简单拼接）
            formatted_prompt = f"System: {self.system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
        
        return formatted_prompt

    def batch_generate(self, batch_prompts):
        """批量生成：OpenAI 或 vLLM"""
        print(f"Generating predictions for {len(batch_prompts)} prompts...")

        formatted_prompts = [self.format_chat_prompt(prompt, self.model_type) for prompt in batch_prompts]

        lora_request = None
        if self.lora_path:
            print(f"Using LoRA adapter '{self.lora_name}' from path '{self.lora_path}' for generation.")
            lora_request = LoRARequest(lora_name=self.lora_name, lora_int_id=1, lora_path=self.lora_path)

        if lora_request:
            outputs = self.llm.generate(formatted_prompts, self.sampling_params, lora_request=lora_request)
        else:
            outputs = self.llm.generate(formatted_prompts, self.sampling_params)

        results = []
        for output in outputs:
            results.append(output.outputs[0].text)
        return results
    
    def parse_text_response(self, response):
        
        predicted_class = None
        predicted_interaction = None

        # 使用正则表达式提取 Answer (类别编号)
        class_match = re.search(r"Class:\s*(\d+)", response)

        if class_match:
            predicted_class = class_match.group(1).strip()

        # 使用正则表达式提取 Interaction type (类别文本)
        interaction_match = re.search(r"Interaction:\s*(.*)", response)

        if interaction_match:
            predicted_interaction = interaction_match.group(1).strip()

        return predicted_class, predicted_interaction
    
    def run_prediction(self, input_file, drugs_file, output_file, kg_path=None, test_limit=None, save_thinking=False):
        """运行完整的预测流程"""
        self.save_thinking = save_thinking
    
        print("Loading data...")
        self.load_data(input_file, drugs_file, kg_path)
    
        if test_limit:
            self.input_file = self.input_file.head(test_limit)
            print(f"Limited to first {test_limit} samples for testing")
        
        print("Creating prompts...")
        prompts = []
        for _, row in self.input_file.iterrows():
            prompt = self.create_user_prompt(
                row['drug1_id'], 
                row['drug2_id'], 
                row_index=row.name
            )
            # print("sample prompt:", row.get('label', None), row.get('class', None))
            # print(prompt)

            prompts.append(prompt)
        
        # 在处理完所有结果后，保存未找到的键信息
        if hasattr(self, 'not_found_keys') and self.not_found_keys:
            self.save_not_found_keys_analysis(output_file)
    
        print("Generating predictions with vLLM...")
        responses = self.batch_generate(prompts)
        
        print("Processing results...")
        
        # 统一的预测结果统计 - 基于所有样本
        all_predictions = []
        correct_results = []  # 保留原有的correct_results用于保存
        failed_results = []   # 保留原有的failed_results用于保存
        
        total_samples = len(self.input_file)
        
        for idx, (_, row) in enumerate(self.input_file.iterrows()):
            actual_class = str(row['class'])
            actual_interaction = row['label']
            
            # 初始化预测记录
            prediction_record = {
                'row_index': idx,
                'drug1_id': row['drug1_id'],
                'drug1_name': row['drug1_name'],
                'drug2_id': row['drug2_id'], 
                'drug2_name': row['drug2_name'],
                'actual_class': actual_class,
                'actual_interaction': actual_interaction,
                'predicted_class': None,
                'predicted_interaction': None,
                'has_response': False,
                'has_valid_json': False,
                'class_correct': False,
                'interaction_correct': False,
                'either_correct': False,
                'error_type': None,
                'question': None,
                'model_output': None
            }
            
            # 检查是否有响应
            if idx >= len(responses):
                prediction_record.update({
                    'error_type': 'No response',
                    'model_output': 'No response generated'
                })
                all_predictions.append(prediction_record)
                
                # 添加到failed_results
                failed_results.append({
                    'row_index': idx,
                    'drug1_id': row['drug1_id'],
                    'drug1_name': row['drug1_name'],
                    'drug2_id': row['drug2_id'],
                    'drug2_name': row['drug2_name'],
                    'actual_class': actual_class,
                    'actual_interaction': actual_interaction,
                    'predicted_class': None,
                    'predicted_interaction': None,
                    'model_output': "No response generated",
                    'error_type': "No response",
                    'question': prompts[idx] if idx < len(prompts) else "Prompt not available"
                })
                continue
                
            response = responses[idx]
            prediction_record.update({
                'has_response': True,
                'question': prompts[idx],
                'model_output': response
            })
            
            # 尝试解析文本响应
            predicted_class, predicted_interaction = self.parse_text_response(response)
            
            if predicted_class is None:
                prediction_record.update({
                    'error_type': 'Parsing failed'
                })
                all_predictions.append(prediction_record)
                
                # 添加到failed_results
                failed_results.append({
                    'row_index': idx,
                    'drug1_id': row['drug1_id'],
                    'drug1_name': row['drug1_name'],
                    'drug2_id': row['drug2_id'],
                    'drug2_name': row['drug2_name'],
                    'actual_class': actual_class,
                    'actual_interaction': actual_interaction,
                    'predicted_class': None,
                    'predicted_interaction': None,
                    'model_output': response,
                    'error_type': "Parsing failed",
                    'question': prompts[idx]
                })
                continue
            
            # 解析成功
            prediction_record['has_valid_json'] = True # Note: This key now means "has valid prediction"
            
            prediction_record.update({
                'predicted_class': str(predicted_class) if predicted_class is not None else None,
                'predicted_interaction': predicted_interaction
            })
            
            # 计算正确性 - 基于所有样本，包括预测失败的
            class_correct = (str(predicted_class) == actual_class) if predicted_class is not None else False
            # Interaction type 评估是可选的，这里主要关注 class
            interaction_correct = (predicted_interaction == actual_interaction) if predicted_interaction is not None else False
            either_correct = class_correct # 主要以类别正确性为准

            prediction_record.update({
                'class_correct': class_correct,
                'interaction_correct': interaction_correct,
                'either_correct': either_correct
            })
            
            # 确定错误类型
            if not either_correct:
                if predicted_class is None:
                    prediction_record['error_type'] = 'No class prediction extracted'
                else:
                    prediction_record['error_type'] = 'Wrong prediction'
            else:
                prediction_record['error_type'] = 'Correct'
            
            all_predictions.append(prediction_record)
            
            # 根据统一标准决定是否加入correct_results（保持原有逻辑）
            if either_correct:
                drug1_smiles = self.drug_info_map.get(row['drug1_id'], {}).get('smiles', '')
                drug2_smiles = self.drug_info_map.get(row['drug2_id'], {}).get('smiles', '')
                
                correct_results.append({
                    'id': idx,
                    'question': prompts[idx],
                    'drug1_smiles': drug1_smiles,
                    'drug2_smiles': drug2_smiles,
                    'answer': predicted_interaction,
                    'reasoning': response, # 将完整输出作为reasoning
                    'label': predicted_class
                })
            else:
                # 添加到failed_results
                failed_results.append({
                    'row_index': idx,
                    'drug1_id': row['drug1_id'],
                    'drug1_name': row['drug1_name'],
                    'drug2_id': row['drug2_id'],
                    'drug2_name': row['drug2_name'],
                    'actual_class': actual_class,
                    'actual_interaction': actual_interaction,
                    'predicted_class': predicted_class,
                    'predicted_interaction': predicted_interaction,
                    'model_output': response,
                    'error_type': prediction_record['error_type'],
                    'question': prompts[idx]
                })
        
        # 转换为DataFrame便于分析
        predictions_df = pd.DataFrame(all_predictions)
        
        # 计算性能指标 - 基于所有样本
        print("\n=== Performance Metrics (Based on ALL Samples) ===")
        
        # 1. 基础统计
        print(f"Total samples: {total_samples}")
        print(f"Valid responses: {predictions_df['has_response'].sum()}")
        print(f"Valid JSON: {predictions_df['has_valid_json'].sum()}")
        print(f"Valid class predictions: {predictions_df['predicted_class'].notna().sum()}")
        
        # 2. 基于所有样本的准确率计算
        class_accuracy = predictions_df['class_correct'].sum() / total_samples
        interaction_accuracy = predictions_df['interaction_correct'].sum() / total_samples
        either_accuracy = predictions_df['either_correct'].sum() / total_samples
        
        print(f"Class Accuracy (all samples): {class_accuracy:.4f} ({predictions_df['class_correct'].sum()}/{total_samples})")
        print(f"Interaction Accuracy (all samples): {interaction_accuracy:.4f} ({predictions_df['interaction_correct'].sum()}/{total_samples})")
        print(f"Either Correct Accuracy (all samples): {either_accuracy:.4f} ({predictions_df['either_correct'].sum()}/{total_samples})")
        
        # 3. 错误类型统计
        print("\n=== Error Type Analysis ===")
        error_counts = predictions_df['error_type'].value_counts()
        print("Error type distribution:")
        for error_type, count in error_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {error_type}: {count} ({percentage:.2f}%)")
        
        # 4. 传统分类指标 - 基于所有样本
        print(f"\n=== Classification Metrics (Based on ALL {total_samples} samples) ===")
        
        # 为所有样本构建预测和真实标签列表
        # 对于没有预测的样本，使用一个特殊的标签（如"NO_PREDICTION"）
        all_actual_classes = []
        all_predicted_classes = []
        
        for _, pred in predictions_df.iterrows():
            all_actual_classes.append(pred['actual_class'])
            
            # 如果没有预测类别，使用特殊标记
            if pred['predicted_class'] is not None:
                all_predicted_classes.append(pred['predicted_class'])
            else:
                all_predicted_classes.append("NO_PREDICTION")  # 特殊标记表示无预测
        
        # 获取所有真实存在的类别列表 (去重并排序)
        unique_labels = sorted(list(set(all_actual_classes)))

        # 计算基于所有样本的分类指标
        precision_macro = precision_score(all_actual_classes, all_predicted_classes, labels=unique_labels, average='macro', zero_division=0)
        recall_macro = recall_score(all_actual_classes, all_predicted_classes, labels=unique_labels,average='macro', zero_division=0)
        f1_macro = f1_score(all_actual_classes, all_predicted_classes, labels=unique_labels,average='macro', zero_division=0)
        
        precision_micro = precision_score(all_actual_classes, all_predicted_classes, labels=unique_labels, average='micro', zero_division=0)
        recall_micro = recall_score(all_actual_classes, all_predicted_classes, labels=unique_labels, average='micro', zero_division=0)
        f1_micro = f1_score(all_actual_classes, all_predicted_classes, labels=unique_labels, average='micro', zero_division=0)
        
        # 基于所有样本的准确率（与class_accuracy相同）
        overall_class_accuracy = accuracy_score(all_actual_classes, all_predicted_classes)
        
        print(f"Overall Class Accuracy: {overall_class_accuracy:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"Precision (Micro): {precision_micro:.4f}")
        print(f"Recall (Micro): {recall_micro:.4f}")
        print(f"F1-Score (Micro): {f1_micro:.4f}")
        
        # 5. 仅基于有效预测的分类指标（用于对比）
        valid_class_predictions = predictions_df[predictions_df['predicted_class'].notna()]
        
        if len(valid_class_predictions) > 0:
            print(f"\n=== Classification Metrics (Based on {len(valid_class_predictions)} samples with valid predictions - for comparison) ===")
            
            valid_actual_classes = valid_class_predictions['actual_class'].tolist()
            valid_predicted_classes = valid_class_predictions['predicted_class'].tolist()
            
            valid_precision_macro = precision_score(valid_actual_classes, valid_predicted_classes, labels=unique_labels, average='macro', zero_division=0)
            valid_recall_macro = recall_score(valid_actual_classes, valid_predicted_classes, labels=unique_labels, average='macro', zero_division=0)
            valid_f1_macro = f1_score(valid_actual_classes, valid_predicted_classes, labels=unique_labels,average='macro', zero_division=0)
            
            valid_class_accuracy = accuracy_score(valid_actual_classes, valid_predicted_classes)
            
            print(f"Valid Predictions Class Accuracy: {valid_class_accuracy:.4f}")
            print(f"Valid Predictions Precision (Macro): {valid_precision_macro:.4f}")
            print(f"Valid Predictions Recall (Macro): {valid_recall_macro:.4f}")
            print(f"Valid Predictions F1-Score (Macro): {valid_f1_macro:.4f}")
        
        # 保存详细指标
        metrics_output = output_file.replace('.csv', '_metrics.txt')
        with open(metrics_output, 'w') as f:
            f.write("=== Performance Metrics ===\n\n")
            
            f.write("=== Overall Performance (All Samples) ===\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Valid responses: {predictions_df['has_response'].sum()}\n")
            f.write(f"Valid JSON: {predictions_df['has_valid_json'].sum()}\n")
            f.write(f"Valid class predictions: {predictions_df['predicted_class'].notna().sum()}\n\n")
            
            f.write(f"Class Accuracy (all samples): {class_accuracy:.4f} ({predictions_df['class_correct'].sum()}/{total_samples})\n")
            f.write(f"Interaction Accuracy (all samples): {interaction_accuracy:.4f} ({predictions_df['interaction_correct'].sum()}/{total_samples})\n")
            f.write(f"Either Correct Accuracy (all samples): {either_accuracy:.4f} ({predictions_df['either_correct'].sum()}/{total_samples})\n\n")
            
            f.write("=== Error Type Distribution ===\n")
            for error_type, count in error_counts.items():
                percentage = (count / total_samples) * 100
                f.write(f"{error_type}: {count} ({percentage:.2f}%)\n")
            f.write("\n")
            
            f.write("=== Classification Metrics (All Samples) ===\n")
            f.write(f"Overall Class Accuracy: {overall_class_accuracy:.4f}\n")
            f.write(f"Precision (Macro): {precision_macro:.4f}\n")
            f.write(f"Recall (Macro): {recall_macro:.4f}\n")
            f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
            f.write(f"Precision (Micro): {precision_micro:.4f}\n")
            f.write(f"Recall (Micro): {recall_micro:.4f}\n")
            f.write(f"F1-Score (Micro): {f1_micro:.4f}\n\n")
            
            f.write("Detailed Classification Report (based on all samples):\n")
            f.write(classification_report(all_actual_classes, all_predicted_classes, zero_division=0))
            f.write("\n")
            
            if len(valid_class_predictions) > 0:
                f.write("=== Classification Metrics (Valid Predictions Only - for comparison) ===\n")
                f.write(f"Valid Predictions Class Accuracy: {valid_class_accuracy:.4f}\n")
                f.write(f"Valid Predictions Precision (Macro): {valid_precision_macro:.4f}\n")
                f.write(f"Valid Predictions Recall (Macro): {valid_recall_macro:.4f}\n")
                f.write(f"Valid Predictions F1-Score (Macro): {valid_f1_macro:.4f}\n\n")
                
                f.write("Detailed Classification Report (based on valid predictions only):\n")
                f.write(classification_report(valid_actual_classes, valid_predicted_classes, zero_division=0))
        
        print(f"Performance metrics saved to {metrics_output}")
        
        # 保存详细预测结果
        detailed_output = output_file.replace('.csv', '_detailed_predictions.csv')
        predictions_df.to_csv(detailed_output, index=False)
        print(f"Detailed predictions saved to {detailed_output}")
        
        # 保存结果（保持原有逻辑）
        if correct_results:
            results_df = pd.DataFrame(correct_results)
            results_df.to_csv(output_file, index=False)
            print(f"Saved {len(correct_results)} correct predictions to {output_file}")
        else:
            print("No correct predictions found.")
        
        # 保存失败结果
        if failed_results:
            failed_details_df = pd.DataFrame(failed_results)
            failed_details_output = output_file.replace('.csv', '_failed_details.csv')
            failed_details_df.to_csv(failed_details_output, index=False)
            print(f"Saved {len(failed_results)} failed prediction details to {failed_details_output}")
        
        # 验证评估过程和评估结束后的一致性
        print(f"\n=== Consistency Check ===")
        print(f"len(correct_results): {len(correct_results)}")
        print(f"either_correct count: {predictions_df['either_correct'].sum()}")
        print(f"Consistency: {len(correct_results) == predictions_df['either_correct'].sum()}")
        
        # 返回基于所有样本的统计
        return len(correct_results), total_samples, list(predictions_df[~predictions_df['either_correct']].index)
   
def main():
    # 配置文件路径（切换为 Baichuan-M2-32B）
    model_type = "qwen"
    model_path = "/data1/gaoshang/Project/LLMs/Qwen/Qwen3/4B_Instruct_2507"
    # 在这里指定你的LoRA权重所在的目录路径
    lora_path = "/data1/gaoshang/Project/Drug_Drug_Interactions/experiments/Qwen3/4B_Instruct_2507/S11_LoRA_8_32_FSDP_Sample/lora_test"
    drugs_file = "/data1/gaoshang/Project/Drug_Drug_Interactions/dataset/drug_reason/drugbank/drugs_info_keep_valid.csv"
    # kg_path_file = "/data/gaoshang/Project/Drug_Drug_Interactions/dataset/drug_reason/BioKG_HetioNet/biokg_info_v2.json"
    kg_path_file = "/data1/gaoshang/Project/Drug_Drug_Interactions/dataset/drug_reason/BioKG_HetioNet/DrugBank/S1/1/drugbank_test_add_reverse_k-paths-ddi_l15_p20.json"
    input_file = "/data1/gaoshang/Project/Drug_Drug_Interactions/dataset/drug_reason/drugbank/S1/1/test_full.csv"
    output_file = "/data1/gaoshang/Project/Drug_Drug_Interactions/experiments/Qwen3/4B_Instruct_2507/S11_LoRA_8_32_FSDP_Sample/test_results/s1_1_test_full_qwen3_v23_role_1E_l15p20.csv"

    try:
        print("Initializing vLLM predictor...")
        predictor = DrugInteractionPredictor(
            model_path, 
            model_type=model_type, 
            lora_path=lora_path, # <--- 将LoRA路径传递给构造函数
            save_thinking=False,
            backend=None,  # 默认：开启 ENABLE_GPT_5_1_CODEX_MAX_FOR_ALL_CLIENTS=1 时自动用 openai
        )
        print("vLLM predictor initialized successfully!")

        # 运行预测
        correct_count, total_count, failed_indices = predictor.run_prediction(
            input_file, drugs_file, output_file, kg_path=kg_path_file, test_limit=None)
    
        print(f"Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.4f}")
        print(f"Failed predictions: {len(failed_indices)} samples")

        if failed_indices and predictor.not_found_keys:
            missing_kg_indices = {item['row_index'] for item in predictor.not_found_keys}
            failed_prediction_indices = set(failed_indices)
            
            failed_with_missing_kg = failed_prediction_indices.intersection(missing_kg_indices)
            
            count = len(failed_with_missing_kg)
            percentage = (count / len(failed_indices)) * 100 if failed_indices else 0
            
            print(f"\n--- Analysis of Failed Predictions ---")
            print(f"Total failed predictions: {len(failed_indices)}")
            print(f"Total samples with missing KG path: {len(missing_kg_indices)}")
            print(f"Failed predictions that also have a missing KG path: {count} ({percentage:.2f}%)")


    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Script started...")
    main()
    print("Script finished.")

