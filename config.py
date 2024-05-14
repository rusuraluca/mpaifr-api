class CONFIG(object):
    """
    CONFIG is a Singleton class that defines one instance of a list with the configuration that the API should use.
    """
    _instance = None

    def __init__(self):
        """
        Raise an exception in __init__() to make normal object instantiation impossible.
        """
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        """
        Callers are instructed to use the instance() class method,
        which creates the object once and returns the object.
        :return: list of layers
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance = {
                'model_name': 'multitask_dal', #singletask, multitask, multitask_dal
                'dataset': 'small', #big, small
                'model_weights': '',
                'margin_loss': 'cosface', #cosface, arcface
            }
        return cls._instance
