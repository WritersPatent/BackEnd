import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union, Optional
import json

class DataLoader:
    def __init__(self):
        self.supported_extensions = ['.csv', '.xlsx', '.txt', '.json']
    
    def load_data(self, file_path: str, text_col: str = None, label_col: str = None) -> Tuple[List[str], List[int]]:
        """
        다양한 형식의 데이터 파일을 로드하여 텍스트와 레이블을 반환합니다.
        
        Args:
            file_path (str): 데이터 파일 경로
            text_col (str): 텍스트 컬럼명 (기본값: None, 자동 감지 시도)
            label_col (str): 레이블 컬럼명 (기본값: None, 자동 감지 시도)
            
        Returns:
            Tuple[List[str], List[int]]: (텍스트 리스트, 레이블 리스트)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
        if file_path.suffix not in self.supported_extensions:
            raise ValueError(f"지원하지 않는 파일 형식입니다. 지원되는 형식: {self.supported_extensions}")
            
        try:
            if file_path.suffix == '.csv':
                return self._load_csv(file_path, text_col, label_col)
            elif file_path.suffix == '.xlsx':
                return self._load_excel(file_path, text_col, label_col)
            elif file_path.suffix == '.txt':
                return self._load_txt(file_path)
            elif file_path.suffix == '.json':
                return self._load_json(file_path)
        except Exception as e:
            raise Exception(f"데이터 로드 중 오류 발생: {str(e)}")
    
    def _detect_columns(self, df: pd.DataFrame, text_col: Optional[str] = None, 
                       label_col: Optional[str] = None) -> Tuple[str, str]:
        """컬럼명을 자동으로 감지합니다."""
        # 텍스트 컬럼 감지
        text_column_candidates = ['text', 'content', 'sentence', 'document', '텍스트', '내용', '문장']
        if text_col is None:
            for candidate in text_column_candidates:
                if candidate in df.columns:
                    text_col = candidate
                    break
            if text_col is None:
                # 첫 번째 문자열 타입 컬럼을 텍스트 컬럼으로 가정
                string_columns = df.select_dtypes(include=['object']).columns
                if len(string_columns) > 0:
                    text_col = string_columns[0]
                else:
                    raise ValueError("텍스트 컬럼을 찾을 수 없습니다.")
        
        # 레이블 컬럼 감지
        label_column_candidates = ['label', 'class', 'category', 'target', '레이블', '분류', '카테고리']
        if label_col is None:
            for candidate in label_column_candidates:
                if candidate in df.columns:
                    label_col = candidate
                    break
            if label_col is None:
                # 마지막 숫자 타입 컬럼을 레이블 컬럼으로 가정
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_columns) > 0:
                    label_col = numeric_columns[-1]
                else:
                    raise ValueError("레이블 컬럼을 찾을 수 없습니다.")
        
        return text_col, label_col
    
    def _load_csv(self, file_path: Path, text_col: Optional[str] = None, 
                  label_col: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """CSV 파일 로드"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp949')
            
        text_col, label_col = self._detect_columns(df, text_col, label_col)
        print(f"사용된 컬럼 - 텍스트: {text_col}, 레이블: {label_col}")
        
        return df[text_col].tolist(), df[label_col].astype(int).tolist()
    
    def _load_excel(self, file_path: Path, text_col: Optional[str] = None, 
                   label_col: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """Excel 파일 로드"""
        df = pd.read_excel(file_path)
        text_col, label_col = self._detect_columns(df, text_col, label_col)
        print(f"사용된 컬럼 - 텍스트: {text_col}, 레이블: {label_col}")
        
        return df[text_col].tolist(), df[label_col].astype(int).tolist()
    
    def _load_txt(self, file_path: Path) -> Tuple[List[str], List[int]]:
        """
        TXT 파일 로드
        형식: text\tlabel 또는 text,label
        """
        texts, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # 빈 줄 무시
                    try:
                        if '\t' in line:
                            text, label = line.strip().split('\t')
                        else:
                            text, label = line.strip().split(',')
                        texts.append(text)
                        labels.append(int(label))
                    except ValueError:
                        print(f"경고: 잘못된 형식의 라인 무시됨: {line.strip()}")
        return texts, labels
    
    def _load_json(self, file_path: Path) -> Tuple[List[str], List[int]]:
        """
        JSON 파일 로드
        형식: [{"text": "텍스트", "label": 레이블}, ...]
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                # 리스트의 딕셔너리 형식
                texts = [item.get('text', item.get('content', '')) for item in data]
                labels = [int(item.get('label', item.get('class', 0))) for item in data]
            else:
                # 단순 리스트 형식 [텍스트, 레이블, 텍스트, 레이블, ...]
                texts = [data[i] for i in range(0, len(data), 2)]
                labels = [int(data[i]) for i in range(1, len(data), 2)]
        else:
            raise ValueError("지원하지 않는 JSON 형식입니다.")
            
        return texts, labels

    def save_data(self, texts: List[str], labels: List[int], 
                  file_path: str, format: str = 'csv') -> None:
        """데이터를 지정된 형식으로 저장"""
        data = {'text': texts, 'label': labels}
        df = pd.DataFrame(data)
        
        if format == 'csv':
            df.to_csv(file_path, index=False, encoding='utf-8')
        elif format == 'xlsx':
            df.to_excel(file_path, index=False)
        elif format == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                for text, label in zip(texts, labels):
                    f.write(f"{text}\t{label}\n")
        elif format == 'json':
            data = [{'text': t, 'label': l} for t, l in zip(texts, labels)]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"지원하지 않는 저장 형식입니다: {format}") 