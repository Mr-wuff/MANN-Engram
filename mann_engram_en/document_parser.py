import os
import re
import json
from typing import List
import pandas as pd
import PyPDF2
import docx

class DocumentParser:
    """Handles various file types and chunks them into semantic text blocks."""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Simple word-based sliding window chunking."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    @staticmethod
    def parse_file(file_path: str) -> List[str]:
        """
        Reads a file based on its extension and returns chunked text.
        Supported formats: txt, md, pdf, docx, xlsx, xls, csv, json.
        """
        ext = os.path.splitext(file_path)[1].lower()
        extracted_text = ""
        
        try:
            if ext in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                    
            elif ext == '.pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    
            elif ext == '.docx':
                doc = docx.Document(file_path)
                extracted_text = "\n".join([para.text for para in doc.paragraphs])
                
            elif ext in ['.xlsx', '.xls', '.csv']:
                df = pd.read_csv(file_path) if ext == '.csv' else pd.read_excel(file_path)
                # Convert rows to string representation
                extracted_text = "\n".join(df.astype(str).apply(lambda x: ' | '.join(x), axis=1))
                
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    extracted_text = json.dumps(data, indent=2, ensure_ascii=False)
            else:
                print(f"[MANN-Engram Warning] Unsupported file format: {ext}")
                return []
                
            # Clean and apply sliding window chunking
            extracted_text = re.sub(r'\n+', '\n', extracted_text).strip()
            return DocumentParser.chunk_text(extracted_text)
            
        except Exception as e:
            print(f"[MANN-Engram Error] Failed to parse {file_path}: {str(e)}")
            return []