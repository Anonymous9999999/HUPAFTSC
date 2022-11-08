import itertools


class PretrainHPConfig:
    def __init__(self,
                 default_hp,
                 hp_range,
                 is_grid_search=False):
        assert default_hp.keys() == hp_range.keys()
        self._default_hp = default_hp
        self.hp_range = hp_range
        self.hp_grid = list(itertools.product(*self.hp_range.values()))
        self.is_grid_search = is_grid_search

    @property
    def grid_search_total(self):
        return len(self.hp_grid)

    @property
    def default_hp(self):
        return self._default_hp

    def get_curr_hp(self, grid_search_index):
        if not self.is_grid_search:
            return self.default_hp
        else:
            return dict(zip(self.hp_range.keys(), self.hp_grid[grid_search_index]))
