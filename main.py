import random

class RollNumberPicker:
    def __init__(self, roll_numbers):
        self.roll_numbers = roll_numbers

    def pick_random_roll(self):
        if not self.roll_numbers:
            return None
        return random.choice(self.roll_numbers)

    def pick_multiple_rolls(self, times):
        rolls = []
        for _ in range(times):
            rolls.append(self.pick_random_roll())
        return rolls


roll_numbers_list = input("Enter roll numbers separated by commas: ").split(',')
# Convert input to integers
roll_numbers_list = [int(roll.strip()) for roll in roll_numbers_list]


picker = RollNumberPicker(roll_numbers_list)

while True:
    times_to_pick = int(input("How many times would you like to pick a roll number? "))
    multiple_random_rolls = picker.pick_multiple_rolls(times_to_pick)
    print(f"Randomly selected roll numbers ({times_to_pick} times): {multiple_random_rolls}")
    continue_prompt = input("Would you like to pick again? (yes/no): ").strip().lower()
    if continue_prompt != 'yes':
        print("Exiting the program.")
        break

