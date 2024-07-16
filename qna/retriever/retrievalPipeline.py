import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

import os
import tarfile
import torch
import time

from interface import retrievalInterface

from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from haystack.utils import convert_files_to_docs
from haystack.nodes import DensePassageRetriever, TfidfRetriever, BM25Retriever, PreProcessor

from transformers import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer

def set_path_name_ext(org_path) :

	path, files = os.path.split(org_path) #- 경로와 파일명을 분리

	return path

def get_available_memory() :

    gpu_count = torch.cuda.device_count()	
    gpu_mem_info = {}
    gpu_device_info = {}

    for gpu_id in range(gpu_count) :

        gpu_mem_info[gpu_id] = {'total' : 0, 'available' : 0, 'used' : 0}
        gpu_device_info[gpu_id] = torch.cuda.get_device_properties(gpu_id)

    for gpu_id in range(gpu_count) :

        torch.cuda.set_device(gpu_id)
        (available, max) = torch.cuda.mem_get_info()
        used = max - available
        gpu_mem_info[gpu_id]['total'] = max / 1024 / 1024
        gpu_mem_info[gpu_id]['available'] = available / 1024 / 1024
        gpu_mem_info[gpu_id]['used'] = used / 1024 / 1024

    return (gpu_count, gpu_mem_info)


def get_max_available_mem_device() :

    gpu_cnt,mem_info = get_available_memory()
    return_gpu_id = 0
    return_mem_available = 0

    for gpu_id in range(gpu_cnt) :
        if mem_info[gpu_id]['available'] > return_mem_available :
            return_gpu_id = gpu_id
            return_mem_available=mem_info[gpu_id]['available']

    return (return_gpu_id, return_mem_available)

def set_service_gpu() :
		
    gpus = []
    target_gpu, available_memory = get_max_available_mem_device()
    gpus.append(target_gpu)

    global target_device

    if len(gpus) > 0 : 

        target_device = 'cuda:{}'.format(target_gpu)
    else:
        
        target_device = 'cpu'

    return target_device


class RetrievalPipeline(retrievalInterface) :
     
    def __init__(self, model_dict) :

        pass
     
    def load_document(self, load_document_param) :

        self.device = set_service_gpu()
         
        load_document_start = time.time()

        logger.info("load_document start")

        if load_document_param.split_paragraphs == 'True' :
            load_document_param.split_paragraphs = True
        else :
            load_document_param.split_paragraphs = False

        docs = convert_files_to_docs(dir_path = load_document_param.filePath, split_paragraphs = load_document_param.split_paragraphs)
        

        if load_document_param.split_by == 'word' :
            boundary = True
        else :
            boundary = False

        preprocessor = PreProcessor(clean_empty_lines = True, clean_whitespace = True, clean_header_footer = False,
                                    split_by = load_document_param.split_by, split_length = int(load_document_param.split_length), 
                                    split_respect_sentence_boundary = boundary)

        passage_doc = preprocessor.process(docs)
        
        TFIDFDocument_Store = InMemoryDocumentStore(use_bm25 = True)
        BM25Document_Store = InMemoryDocumentStore(use_bm25 = True)

        TFIDF_Retriever = TfidfRetriever(document_store = TFIDFDocument_Store)
        TFIDFDocument_Store.delete_documents()
        TFIDFDocument_Store.write_documents(passage_doc)

        BM25_Retriever = BM25Retriever(document_store = BM25Document_Store)
        BM25Document_Store.delete_documents()
        BM25Document_Store.write_documents(passage_doc)

        names = []

        for key, value in self.model_dict.items() :

            if os.path.exists('faiss_document_store.db') :
                os.remove('faiss_document_store.db')
            else :
                pass

            folder_path = set_path_name_ext(value)
            name = f"{key}_Retriever"

            names.append(name)

            logger.info("name")
            logger.info(name)

            if len(os.listdir(folder_path)) == 1 :
                
                try :

                    logger.info(value)

                    tar = tarfile.open(value) 
                    tar.extractall(path = folder_path)
                    tar.close()

                except Exception as e :
                    
                    logger.info(f"압축 해제 실패 : {e}")

            else :
                
                pass     

            FAISSDocument_Store = FAISSDocumentStore(faiss_index_factory_str = "Flat")

            query_path = os.path.join(folder_path, "query_encoder")
            passage_path = os.path.join(folder_path, "passage_encoder")

            DPR_Retriever = DensePassageRetriever(document_store = FAISSDocument_Store,
                                                    query_embedding_model  = query_path,
                                                    passage_embedding_model = passage_path,
                                                    max_seq_len_query = 128, max_seq_len_passage = 512,
                                                    use_fast_tokenizers = False,
                                                    use_gpu = True, devices = [self.device])
            
            DPR_Retriever.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_path)
            DPR_Retriever.processor.passage_tokenizer = DPRContextEncoderTokenizer.from_pretrained(passage_path)
            DPR_Retriever.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_path, do_lower_case = False)
            DPR_Retriever.processor.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_path, do_lower_case = False)
            FAISSDocument_Store.delete_documents()
            FAISSDocument_Store.write_documents(passage_doc)
            FAISSDocument_Store.update_embeddings(retriever = DPR_Retriever)        

            globals()[name] = DPR_Retriever

            logger.info("load_document time")
            load_document_time = time.time() - load_document_start
            logger.info(load_document_time)

        return TFIDF_Retriever, BM25_Retriever, names
    

    def get_global_var(self, names, retriever_algorithm) :

        for name in names :

            if name.split("_")[0] in retriever_algorithm :

                return globals().get(name)

    def retriever(self, TFIDF_Retriever, BM25_Retriever, DPR_Retriever, retriever_param) :

        retriever_start = time.time()

        if retriever_param.retriever_algorithm == 'tf-idf' :
            candidate_document = TFIDF_Retriever.retrieve(query = retriever_param.query, top_k = int(retriever_param.top_k))
        
        elif retriever_param.retriever_algorithm == 'bm25' :
            candidate_document = BM25_Retriever.retrieve(query = retriever_param.query, top_k = int(retriever_param.top_k))

        else :

            candidate_document = DPR_Retriever.retrieve(query = retriever_param.query, top_k = int(retriever_param.top_k))
                    

        logger.info("retriever time")
        retriever_time = time.time() - retriever_start
        logger.info(retriever_time)

        return candidate_document

    def __call__(self) :

        TFIDF_Retriever, BM25_Retriever, names = load_document(doc_dir)

        DPR_Retriever = get_global_var(names)

        candidate_document = retriever(TFIDF_Retriever, BM25_Retriever, names)

        return candidate_document
