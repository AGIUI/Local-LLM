def run(number, min_value, max_value, step):
    if number < min_value:
        number = min_value
    elif number > max_value:
        number = max_value
    scaled_number = (number - min_value) / (max_value - min_value)
    return (scaled_number,)

print(run(1,0,2,0))

print(run(122222222222222,0,0.1,0))
