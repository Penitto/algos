import random


def bubble_sort(arr: list):

    """
    Попарное сравнение каждых 2х последовательных значений.
    Принцип: поставить самый максимальный элемент в конец

    O(n ^ 2) complexity
    """

    for i in range(len(arr)):

        # До конца идти не нужно, потому что там уже отсортированно
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr


def selection_sort(arr: list):

    """
    Поиск минимального элемента в неотсортированной части массива
    и установка его в нужном месте в отсортированной части

    O(n ^ 2) complexity
    """

    for i in range(len(arr) - 1):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j

        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]

    return arr


def insert_sort(arr: list):

    """
    Комбинация bubble и selection.
    Аналогия с игральными картами

    O (n ^ 2) complexity
    """

    for i in range(1, len(arr)):
        cur = arr[i]
        j = i - 1
        while j >= 0:
            if cur < arr[j]:
                arr[j+1], arr[j] = arr[j], cur
                j -= 1
            else:
                break
    return arr


def quick_sort(arr: list):

    """
    Выбор значения, с которым будет происходить сравнение.
    Деление на меньше и больше него и склеивание

    O(n * log(n)) mean complexity
    O(n ^ 2) worst complexity
    """

    if len(arr) <= 1:
        return arr
    else:
        pivot = random.choice(arr)
        less = [x for x in arr if x < pivot]
        greater = [x for x in arr if x >= pivot]
        return quick_sort(less) + quick_sort(greater)


def merge_sort(arr: list):

    """
    Разбиение пополам и сортировка внутри

    O(n * log(n)) complexity
    """
    if len(arr) <= 1:
        return arr
    else:
        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        return merge(left, right)


def merge(left: list, right: list):

    arr = []

    while len(left) > 0 and len(right) > 0:

        # Если в левом массиве первый элемент меньше
        # первого элемента в правом
        if left[0] <= right[0]:
            arr.append(left[0])
            left = left[1:]
        else:
            arr.append(right[0])
            right = right[1:]

    # Присоединить то, что осталось
    if len(left) > 0:
        arr += left
    if len(right) > 0:
        arr += right
    return arr
