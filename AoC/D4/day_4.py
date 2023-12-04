with open("input.txt") as file:
    input_list = file.read().split("\n")

summ = 0
card_list = [1] * len(input_list)
for index, card in enumerate(input_list):
    winning_numbers, numbers = card.split(":")[1].split("|")
    winning_numbers = winning_numbers.split()
    numbers = numbers.split()
    prize_dict = {}
    for nb in winning_numbers:
        prize_dict[nb] = 0
    for nb in numbers:
        if prize_dict.get(nb) is not None:
            prize_dict[nb] += 1
    value = 0
    for _, val in prize_dict.items():
        value += val
    if value != 0:
        first_index = min(index + 1, len(input_list))
        last_index = min(index + 1 + value, len(input_list))
        card_list[first_index:last_index] = list(map(lambda x: x + card_list[index], card_list[first_index:last_index]))
print(card_list)
print(sum(card_list))
