import os

class DataLoader():
    def __init__(self, args):
        self.verbose = args.verbose
        self.input = args.input
        self.output = args.output
        self.files = []

        if not os.path.exists(self.input):
            print("Problem loading data: Input folder does not exist.")
            raise FileExistsError
        if not os.path.exists(self.output):
            os.makedirs(self.output)
            os.makedirs(os.path.join(self.output, "plots"))

        if self.verbose:
            print(f"Dataloader instantiated.\nInput folder is {self.input}\nOutput folder is {self.output}")

    def collect(self, count=10):
        for file in os.listdir(self.input)[:count]:
            filename = os.fsdecode(file)
            if filename.endswith(".wav"):
                self.files.append(os.path.join(self.input, filename))

                if self.verbose:
                    print('\033[K' + f"[{len(self.files):03}/{count:03}] {filename} added", end='\r')
        print()

    def load_random(self, count=100):
        print(f"randomly sampling {count} files")
        return
