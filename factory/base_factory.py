from abc import ABC, abstractmethod

class BaseFactory(ABC):
    '''
    The base factory class that all factories should inherit from.
    '''

    @classmethod
    @abstractmethod
    def create(cls, **kwargs):
        ...