import random
filename = "many_student_ids.txt"
class RollPicker:
    def __init__(self, roll_numbers):
        self.roll_numbers = roll_numbers
    def pick_random_roll(self):
        if not self.roll_numbers:
            return None
        selected_roll = random.choice(self.roll_numbers)
        self.roll_numbers.remove(selected_roll)
        return selected_roll
try:
    with open(filename, "r") as file:
        stus_ids = [line.strip() for line in file]
    select_stus = []
    not_select_stus = stus_ids.copy()
    if not_select_stus:
        print("Students who haven't been selected yet:")
        for roll in not_select_stus:
            print(roll)
        picker = RollPicker(not_select_stus)
        while picker.roll_numbers:
            picked_roll = picker.pick_random_roll()
            if picked_roll:
                print(f"Randomly selected roll number: {picked_roll}")
                select_stus.append(picked_roll)
        print("All students have been picked.")
        not_select_stus = stus_ids.copy()
        print("List has been reset. Current students:", not_select_stus)
    else:
        print("No students left who haven't been selected.")
except FileNotFoundError:
    print(f"The file '{filename}' does not exist.")
