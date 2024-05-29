from loader import File_geojson


class test_model:
    def __init__(self, datapath: list) -> None:

        self.label_dict = {
            0: "Picea abies",
            #    1: "Fagus sylvatica",
            #    2: "Pinus sylvestris",
            3: "Quercus robur",
            4: "Betula pendula",
            5: "Quercus petraea",
            6: "Fraxinus excelsior",
            7: "Acer pseudoplatanus",
            8: "Sorbus aucuparia",
            9: "Carpinus betulus",
        }

        self.odr_file = None
        for i in range(10):  # the class number should as input
            tree_file = File_geojson(datapath[i], self.label_dict)
            self.odr_file.append(tree_file)

        pass

    def process(self):
        pass
