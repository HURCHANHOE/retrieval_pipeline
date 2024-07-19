# retrieval_pipeline
![image](https://github.com/user-attachments/assets/c164e48e-410c-4c6c-af8b-6557e2438acb)

사용한 라이브러리: haystack

파이프라인 설명
  load_document : 문서를 로드하는 메소드
  retriever : 검색 및 리트리버 설정을 처리하는 메소드 검색 알고리즘에 따라 저장소가 상이함

저장 방식 :
  1. FAISSDocumentStore : FAISS는 Facebook AI Research에서 개발한 벡터 검색 라이브러리로, 고차원 벡터를 효과적으로 검색하고 유사성을 계산하는 데 특화
  2. InMemoryDocumentStore : 메모리 내에 문서를 저장하는 간단한 저장소
