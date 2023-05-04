import os
from datasets import load_dataset, Audio

class DataLoader():
    def __init__(self, args):
        self.input = args.input
        self.output = args.output

        if not os.path.exists(self.input):
            print("Problem loading data: Input folder does not exist.")
            raise FileExistsError
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        print(f"Dataloader instantiated.\nInput folder is {self.input}\nOutput folder is {self.output}")

    def load(self, count=390):
        d = load_dataset("Fhrozen/FSD50k", split="test")
        print(d)

    def load_random(self, count=100):
        print(f"randomly sampling {count} files")
        return
