import os

class DataLoader():
    def __init__(self, args):
        self.input = args.input
        self.output = args.output
        self.files = []

        if not os.path.exists(self.input):
            print("Problem loading data: Input folder does not exist.")
            raise FileExistsError
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        print(f"Dataloader instantiated.\nInput folder is {self.input}\nOutput folder is {self.output}")

    def collect(self, count=390):
        while len(self.files) < count:
            i = 1
            for file in os.listdir(self.input):
                filename = os.fsdecode(file)
                if filename.endswith(".wav"):
                    self.files.append(os.path.join(self.input, filename))
                    print(f"{i:03}/{count} {filename} added")
                    i += 1

    def load_random(self, count=100):
        print(f"randomly sampling {count} files")
        return
