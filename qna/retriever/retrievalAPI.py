import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

import torch
import os
import sys
import asyncio
import shutil

from fastapi import FastAPI
from retrievalPipeline import RetrievalPipeline
from pydantic import BaseModel

FM = RetrievalPipeline()

app = FastAPI()

class Doc_dir(BaseModel) :

    filePath : str
    model_path : list
    model_id : list
    split_paragraphs : str
    split_by : str
    split_length : str

@app.post("/005/modelLoad")
async def modelload(doc_dir : Doc_dir) :

    global TFIDF_Retriever
    global BM25_Retriever
    global names
    
    await asyncio.sleep(5)

    model_pairs = zip(doc_dir.model_id, doc_dir.model_path)
    model_dict = dict(model_pairs)

    try :

        logger.info('api model load start')

        TFIDF_Retriever, BM25_Retriever, names = FM.load_document(doc_dir, model_dict)

        logger.info('api model load end')

    except Exception as e :

        logger.info(f'문서를 정상적으로 불러오지 못했습니다. : {e}')

class Utterance(BaseModel) :

    query : str
    top_k : str
    retriever_algorithm : str

@app.post('/005/retriever')
async def retriever(param : Utterance) :

    logger.info('api retriever start')
    logger.info('retriever param')
    logger.info(param)

    if param.retriever_algorithm == 'bm25' or param.retriever_algorithm == 'tf-idf' :

        DPR_Retriever = None

    else : 

        DPR_Retriever = FM.get_global_var(names, param.retriever_algorithm)

    try :

        logger.info('FM.retriever start')

        candidate_document = FM.retriever(TFIDF_Retriever, BM25_Retriever, DPR_Retriever, param)

        logger.info('FM.retriever end')

    except Exception as e :

        logger.info(f'리트리버를 정상적으로 실행하지 못했습니다. : {e}')

    logger.info('api retriever end')
    
    output = ''
    
    for i in range(len(candidate_document)) :
        x = candidate_document[i].content
        output += '\n' + '[response]' + '\n' + x 
    
    logger.info(output)

    return output



# from pydantic import BaseModel

# from retrievalPipeline import RetrievalPipeline

# class Doc_dir(BaseModel) :
#     filePath : str
#     model_path : list
#     model_id : list
#     split_paragraphs : str
#     split_by : str
#     split_length : str

# data = {"filePath" : "/home/pjtl2w01admin/Nolan/qna/data",
#         "model_path" : ["/home/pjtl2w01admin/Nolan/qna/base_model_1/fineModel.tar.gz", "/home/pjtl2w01admin/Nolan/qna/base_model_2/fineModel.tar.gz", "/home/pjtl2w01admin/Nolan/qna/base_model_3/fineModel.tar.gz"],
#         "model_id" : ["DPR1", "DPR2", "DPR3"],
#         "split_paragraphs" : "True",
#         "split_by" : "passage",
#         "split_length" : "1"}

# load_document_param = Doc_dir(**data)

# model_pairs = zip(load_document_param.model_id, load_document_param.model_path)
# model_dict = dict(model_pairs)

# FM = RetrievalPipeline(model_dict)

# TFIDF_Retriever, BM25_Retriever, names = FM.load_document(load_document_param)

# class Utterance(BaseModel) :
#     query : str
#     top_k : str
#     retriever_algorithm : str

# re_data = {"query" : "쏠편한 비상금 대출의 개요는",
#            "top_k" : "1",
#            "retriever_algorithm" : "DPR1"}

# retriever_param = Utterance(**re_data)

# DPR_Retriever = FM.get_global_var(names, retriever_param.retriever_algorithm)

# candidate_document = FM.retriever(TFIDF_Retriever, BM25_Retriever, DPR_Retriever, retriever_param)

# print(candidate_document)
