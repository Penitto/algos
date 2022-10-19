from typing import Union

def recursive_binary_search(array: list, value: Union[int, float], low=0, high=None) -> int:

    """
    Recursive binary search for SORTED list
    """

    # for first iteration
    if high is None:
        high = len(array) - 1

    if high >= low:

        mid = (high + low) // 2

        # if less
        if value < array[mid]:
            return recursive_binary_search(array, value, low, mid - 1)
        
        #if more
        elif value > array[mid]:
            return recursive_binary_search(array, value, mid + 1, high)

        else:
            return mid
    else:
        return -1

def iterative_binary_search(array: list, value:Union[int, float]) -> int:

    """
    Iterative binary search for SORTED list
    """
    pass


