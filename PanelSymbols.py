from itertools import chain

class Symbol:
    def __init__(self, value: int, symbol_type: str, idx: int = None):
        """
        value: integer between 1–9
        symbol_type: "number" or "symbol"
        """
        self.value = value
        self.type = symbol_type.lower()  #  "key" , "number" or "symbol"
        self.idx = idx

    def __repr__(self):
        return f"<Symbol {self.value} ({self.type})>"
    


class PanelSymbols:
    """
    Holds panel definitions.
    Each panel is a dict with keys:
      - keys: list of ints
      - values: list of ints
      - symbols: list of ints
    Panel IDs are strings, e.g. "0", "i1", "l4", "a3".
    """ 

    PANEL_0 = {
        "keys": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "symbols": [
                1, 5, 4, 9, 3, 8, 9, 5, 4, 1, 6, 3, 9, 4, 7, 
                4, 3, 7, 4, 6, 2, 3, 8, 7, 2, 4, 7, 1, 3, 6, 
                1, 7, 2, 8, 4, 5, 1, 9, 6, 3, 5, 9, 2, 6, 9,
                7, 4, 8, 2, 9, 4, 2, 3, 5, 8, 9, 2, 3, 8, 5,
                9, 6, 5, 4, 2, 6, 5, 4, 1, 5, 7, 4, 2, 6, 3,
                4, 8, 9, 6, 1, 7, 3, 2, 5, 3, 1, 9, 5, 1, 2,
                9, 7, 6, 8, 2, 3, 4, 9, 7, 6, 8, 2, 1, 6, 9,
                7, 3, 5, 1, 8, 7, 1, 2, 5, 1, 3, 6, 5, 7, 4
                ]
    }

    PANEL_a3 = {
        "keys": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "symbols": [
                6, 7, 5, 3, 6, 8, 4, 6, 1, 2, 4, 3, 1, 6, 8,
                9, 8, 3, 7, 8, 6, 2, 5, 3, 9, 2, 1, 5, 7, 3,
                8, 6, 2, 1, 6, 5, 4, 8, 7, 2, 8, 6, 3, 5, 1,
                9, 1, 8, 4, 3, 8, 1, 9, 3, 1, 9, 2, 7, 1, 4,
                1, 5, 3, 2, 8, 7, 5, 6, 8, 9, 4, 8, 9, 6, 8,
                9, 2, 6, 9, 7, 5, 1, 9, 2, 3, 5, 9, 4, 3, 9,
                6, 5, 8, 2, 5, 7, 6, 4, 5, 6, 7, 1, 5, 7, 8,
                7, 9, 2, 8, 4, 3, 8, 7, 1, 3, 4, 5, 3, 9, 4,
                ]
    }

    PANEL_a5 = {
        "keys": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "symbols": [
                9, 1, 5, 4, 7, 6, 3, 9, 2, 4, 1, 6, 7, 9, 1,
                7, 5, 9, 7, 5, 9, 2, 8, 4, 1, 7, 2, 9, 8, 4,
                2, 4, 1, 5, 7, 8, 3, 5, 1, 3, 2, 9, 6, 7, 3,
                6, 9, 5, 4, 1, 7, 4, 8, 9, 5, 6, 4, 5, 2, 1,
                4, 6, 9, 8, 7, 5, 3, 6, 5, 8, 3, 6, 8, 1, 3,
                2, 7, 4, 3, 8, 4, 7, 1, 8, 5, 2, 8, 3, 9, 1,
                7, 3, 2, 8, 5, 2, 4, 5, 3, 4, 1, 9, 2, 1, 4,
                6, 9, 7, 6, 3, 7, 1, 3, 2, 1, 9, 2, 4, 8, 2,
                ]
    }

    PANEL_i1 = {
        "keys": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "symbols": [
                5, 6, 1, 9, 8, 4, 1, 7, 2, 9, 4, 2, 1, 5, 3,
                4, 9, 2, 1, 3, 8, 2, 4, 5, 6, 9, 7, 5, 2, 9,
                2, 3, 4, 7, 5, 2, 4, 6, 1, 7, 2, 3, 9, 5, 2,
                8, 2, 9, 1, 8, 9, 2, 4, 7, 9, 6, 8, 5, 7, 6,
                9, 1, 4, 5, 3, 6, 1, 6, 5, 7, 4, 3, 6, 8, 5,
                6, 4, 3, 1, 4, 8, 5, 7, 9, 8, 6, 9, 4, 3, 8,
                9, 3, 5, 2, 7, 9, 8, 3, 2, 9, 7, 3, 5, 4, 1,
                3, 8, 6, 1, 2, 6, 3, 8, 6, 4, 9, 2, 7, 6, 2,
                ]
    }

    PANEL_l4 = {
        "keys": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "symbols": [
                8, 6, 2, 3, 8, 1, 2, 4, 9, 2, 7, 4, 6, 5, 8,
                5, 3, 1, 7, 2, 8, 9, 3, 7, 6, 4, 8, 1, 2, 5,
                8, 9, 7, 4, 8, 1, 2, 7, 8, 2, 9, 5, 6, 3, 9,
                9, 2, 8, 1, 2, 3, 9, 2, 3, 1, 8, 2, 5, 6, 4,
                4, 7, 1, 4, 3, 7, 4, 1, 6, 9, 7, 8, 3, 5, 7,
                1, 4, 8, 6, 4, 5, 2, 6, 9, 3, 4, 6, 2, 7, 3,
                8, 1, 5, 9, 7, 6, 9, 3, 8, 6, 5, 4, 1, 3, 2,
                4, 3, 9, 4, 5, 9, 4, 1, 2, 4, 7, 5, 2, 1, 7,
            ]
    }

    PANEL_l3 = {
        "keys": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "numbers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "symbols": [
                6, 4, 1, 2, 7, 1, 3, 9, 6, 1, 8, 3, 5, 2, 3,
                7, 8, 6, 7, 4, 5, 7, 4, 2, 3, 6, 2, 1, 5, 9,
                5, 6, 9, 2, 5, 7, 4, 1, 5, 8, 9, 1, 3, 8, 5,
                4, 2, 3, 6, 4, 5, 8, 6, 2, 5, 7, 9, 5, 3, 4,
                8, 7, 1, 5, 9, 2, 4, 3, 9, 4, 5, 8, 9, 4, 7,
                3, 1, 9, 2, 4, 7, 5, 4, 6, 8, 4, 3, 6, 7, 5,
                5, 8, 6, 4, 5, 8, 6, 1, 8, 5, 1, 7, 3, 2, 8,
                4, 2, 3, 1, 9, 3, 8, 2, 1, 9, 8, 6, 4, 9, 2,
            ]
    }




    panels = {
        "0": PANEL_0,
        "i1": PANEL_i1,
        "l4": PANEL_l4,
        "l3": PANEL_l3,
        "a3": PANEL_a3,
        "a5": PANEL_a5,
    }

    @staticmethod
    def _get_panel_symbols(panel_name: str):
        """
        Return a list of Symbol objects for the given panel_id
        """
        if panel_name not in PanelSymbols.panels:
            raise ValueError(f"No symbols defined for panel '{panel_name}'")

        panel = PanelSymbols.panels[panel_name]
        symbols = {"keys": [], "numbers": [], "symbols": []}

        # Keys → indices 0–8
        for i, val in enumerate(panel["keys"]):
            symbol = Symbol(val, "key", idx=i)
            symbols["keys"].append(symbol)

        # Values → indices 9–17
        for i, val in enumerate(panel["numbers"]):
            symbol = Symbol(val, "number", idx=9 + i)
            symbols["numbers"].append(symbol)

        # Symbols → indices 18+
        for i, val in enumerate(panel["symbols"]):
            symbol = Symbol(val, "symbol", idx=18 + i)
            symbols["symbols"].append(symbol)

        return symbols
    
    def get_panel_symbols(panel_name: str) -> list[Symbol]:
        """
        Flatten panel dict into a single ordered list of Symbol objects
        matching the ROI order.
        """
        symbols_dict = PanelSymbols._get_panel_symbols(panel_name)
        # The correct logical order: keys → numbers → symbols
        return list(chain.from_iterable(symbols_dict[group] for group in ["keys", "numbers", "symbols"]))
