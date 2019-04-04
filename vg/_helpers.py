def pluralize(noun, count):
    return noun if count == 1 else "{}s".format(noun)


def raise_dimension_error(*input_values):
    messages = [
        "{} {}".format(input_value.ndim, pluralize("dimension", input_value.ndim))
        for input_value in input_values
    ]
    if len(messages) == 1:
        message = messages[0]
    elif len(messages) == 2:
        message = "{} and {}".format(*messages)
    else:
        message = "those inputs"
    raise ValueError("Not sure what to do with {}".format(message))
