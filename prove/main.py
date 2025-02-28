

def action(sample, i):
    first_half = sample[:i]
    second_half = sample[i+2:]
    middle = [sample[i] + sample[i + 1]]
    return first_half + middle + second_half

def main(nums):

    queue = [nums]
    costs = [0]

    solutions = {}

    while len(queue) > 0:
        sample = queue.pop(0)
        cost = costs.pop(0)

        if sample == sample[::-1]:
            key = ''.join(map(str, sample))
            solutions[key] = cost

        if len(sample)>2:
            for i in range(len(sample)-1):
                new_sample = action(sample, i)
                queue.append(new_sample)
                costs.append(cost+1)

    best_key = min(solutions, key=solutions.get)
    return best_key, solutions[best_key]




nums = [1, 4, 3, 4, 3, 1]
result, cost = main(nums)
print(result, cost)