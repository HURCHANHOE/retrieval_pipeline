from abc import ABC, abstractmethod

class retrievalInterface(ABC) :

    @abstractmethod
    def __init__(self) :
        pass

    @abstractmethod
    def load_document(self) :
        pass

    @abstractmethod
    def retriever(self) :
        pass

