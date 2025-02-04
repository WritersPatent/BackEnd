import os
import torch
import wandb
from transformers import BertTokenizerFast, GPT2Tokenizer
import yaml
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from wandb.sdk.wandb_init import init

from data_preprocessing import TextAugmentation, KoreanTextDataset, KoreanGenerationDataset
from model import KoreanTextClassifier, KoreanLanguageGenerator, KoreanLLM
from trainer import ModelTrainer, GenerationTrainer
from data_loader import DataLoader
from utils import clear_gpu_memory, get_gpu_info

# GPU 설정을 위한 환경 변수 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 특정 GPU만 사용하고 싶을 때

# PyTorch가 CuDNN을 사용하도록 설정
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_wandb(config: Dict[str, Any], project_name: str):
    try:
        wandb.login()  # wandb 로그인 추가
        wandb.init(
            project=project_name,
            config=config,
            name=config.get('run_name', 'experiment')
        )
    except Exception as e:
        print(f"Warning: wandb 초기화 실패. 오프라인으로 실행합니다. 에러: {str(e)}")
        # wandb 없이도 실행되도록 처리
        return None

def initialize_model():
    """모델 초기화 함수"""
    try:
        print("\n=== 모델 초기화 시작 ===")
        model = KoreanLLM()
        print("=== 모델 초기화 완료 ===\n")
        return model
    except Exception as e:
        print(f"모델 초기화 중 오류 발생: {str(e)}")
        return None

def interactive_mode(model):
    """대화형 모드 실행 함수"""
    print("\n=== 대화형 모드 시작 ===")
    print("'quit' 또는 'exit'를 입력하면 프로그램이 종료됩니다.")
    
    while True:
        try:
            # 사용자 입력 받기
            instruction = input("\n지시사항을 입력하세요: ").strip()
            
            # 종료 조건 확인
            if instruction.lower() in ['quit', 'exit']:
                print("\n프로그램을 종료합니다.")
                break
            
            if not instruction:
                print("지시사항을 입력해주세요!")
                continue
                
            # 추가 입력 받기
            input_text = input("추가 입력이 필요하다면 입력하세요 (없으면 Enter): ").strip()
            
            # 프롬프트 생성
            prompt = model.format_prompt(instruction, input_text) if input_text else model.format_prompt(instruction)
            
            print("\n=== 텍스트 생성 중... ===")
            
            # 텍스트 생성
            generated_texts = model.generate_text(
                prompt,
                max_length=300,
                temperature=0.7,
                num_return_sequences=1
            )
            
            # 결과 출력
            print("\n=== 생성된 텍스트 ===")
            for i, text in enumerate(generated_texts, 1):
                print(f"\n{text}")
            
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
            print("다시 시도해주세요.")
            continue

def main():
    """메인 함수"""
    try:
        # GPU 정보 출력
        get_gpu_info()
        
        # 모델 초기화
        model = initialize_model()
        if model is None:
            print("모델 초기화 실패. 프로그램을 종료합니다.")
            return
        
        # 대화형 모드 실행
        interactive_mode(model)
        
        # 종료 시 GPU 메모리 정리
        clear_gpu_memory()
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        clear_gpu_memory()
    except Exception as e:
        print(f"\n예상치 못한 오류 발생: {str(e)}")
        clear_gpu_memory()

if __name__ == "__main__":
    main()
