from typing import NamedTuple
from src.stock import Stock

class Pair(NamedTuple):
    stock_one: Stock
    stock_two: Stock