import torch
import torch.nn as nn
from transformers import BertModel, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, PreTrainedTokenizerFast, pipeline
import warnings
warnings.filterwarnings('ignore')

class KoreanTextClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('klue/bert-base')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)

class KoreanLanguageGenerator(nn.Module):
    def __init__(self, model_name='skt/kogpt2-base-v2', dropout=0.1):
        """
        한국어 텍스트 생성을 위한 모델 클래스

        Args:
            model_name (str): 사용할 사전학습 모델 이름
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # 모델 설정 고정
        self.transformer.config.pad_token_id = self.transformer.config.eos_token_id
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass 구현

        Args:
            input_ids (torch.Tensor): 입력 텍스트의 토큰 ID
            attention_mask (torch.Tensor, optional): 어텐션 마스크
            labels (torch.Tensor, optional): 학습을 위한 레이블

        Returns:
            transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    def generate_text(self, 
                     prompt, 
                     tokenizer, 
                     max_length=100, 
                     num_return_sequences=1,
                     temperature=0.7,
                     top_k=50,
                     top_p=0.95,
                     repetition_penalty=1.2,
                     no_repeat_ngram_size=3,
                     **kwargs):
        """
        주어진 프롬프트를 기반으로 텍스트 생성

        Args:
            prompt (str): 생성의 시작점이 될 텍스트
            tokenizer: 토크나이저 객체
            max_length (int): 생성할 최대 토큰 수
            num_return_sequences (int): 생성할 시퀀스 수
            temperature (float): 생성 다양성 조절 (높을수록 다양)
            top_k (int): 다음 토큰 선택시 고려할 상위 k개 토큰
            top_p (float): nucleus sampling 확률
            repetition_penalty (float): 반복 패널티
            no_repeat_ngram_size (int): 반복하지 않을 n-gram 크기
            **kwargs: generate 메소드에 전달할 추가 인자

        Returns:
            List[str]: 생성된 텍스트 리스트
        """
        self.eval()
        encoded_prompt = tokenizer(prompt, return_tensors='pt')
        input_ids = encoded_prompt['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded_prompt['attention_mask'].to(next(self.parameters()).device)
        
        with torch.no_grad():
            outputs = self.transformer.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_texts = []
        for sequence in outputs:
            generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(generated_text)
            
        return generated_texts
    
    def save_model(self, path):
        """모델 저장"""
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        """모델 로드"""
        self.load_state_dict(torch.load(path))
        
    @staticmethod
    def load_from_checkpoint(path, model_name='skt/kogpt2-base-v2'):
        """체크포인트에서 모델 로드"""
        model = KoreanLanguageGenerator(model_name)
        model.load_model(path)
        return model

class KoreanLanguageGeneratorWithPrefix(KoreanLanguageGenerator):
    """프리픽스 튜닝을 지원하는 한국어 생성 모델"""
    
    def __init__(self, model_name='skt/kogpt2-base-v2', prefix_length=20, dropout=0.1):
        super().__init__(model_name, dropout)
        
        # 프리픽스 임베딩 레이어 추가
        self.prefix_length = prefix_length
        self.prefix_tokens = nn.Parameter(
            torch.randn(1, prefix_length, self.transformer.config.hidden_size)
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        # 프리픽스 확장
        prefix_tokens = self.prefix_tokens.expand(batch_size, -1, -1)
        
        # 입력에 프리픽스 적용
        if attention_mask is not None:
            prefix_attention = torch.ones(
                batch_size, self.prefix_length, 
                device=attention_mask.device, 
                dtype=attention_mask.dtype
            )
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=prefix_tokens,
            return_dict=True
        )
        
        return outputs

class KoreanLLM:
    def __init__(self, model_name="beomi/KoAlpaca-Polyglot-5.8B"):
        """
        한국어 언어 모델 초기화
        
        Args:
            model_name (str): 사용할 모델 이름
                - beomi/KoAlpaca-Polyglot-5.8B
                - EleutherAI/polyglot-ko-5.8b
                - nlpai-lab/kullm-polyglot-5.8b-v2
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # GPU 메모리 설정
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            print("\n토크나이저 로딩 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                trust_remote_code=True
            )
            
            print("모델 로딩 중...")
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            print("모델 초기화 완료!")
            
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {str(e)}")
            raise
    
    def generate_text(self, 
                     prompt: str,
                     max_length: int = 200,
                     temperature: float = 0.7,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     num_return_sequences: int = 1,
                     repetition_penalty: float = 1.2) -> list:
        """
        프롬프트를 기반으로 텍스트 생성
        """
        try:
            outputs = self.pipeline(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                return_full_text=False  # 프롬프트 제외하고 생성된 텍스트만 반환
            )
            
            return [output['generated_text'] for output in outputs]
            
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {str(e)}")
            return []
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """
        프롬프트 포맷 지정
        """
        if input_text:
            return f"""### 명령어:
{instruction}

### 입력:
{input_text}

### 응답:"""
        else:
            return f"""### 명령어:
{instruction}

### 응답:"""
