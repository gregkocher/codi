import re


# ==========================================
# 1. The Generator Function (from previous step)
# ==========================================
def generate_consistent_prompts(x, y, z):
    intermediate_val = x + y
    additive_final_component = intermediate_val * z
    multiplicative_final_component = 1 + z

    # Parameter mapping for the 10 templates
    p1 = (x, y, z)
    p2 = (x, y, multiplicative_final_component)
    p3 = (x, y, additive_final_component)
    p4 = (x + 2 * y, y, multiplicative_final_component)
    p5 = (x + 2 * y, y, additive_final_component)
    p6 = (x + y, 1, multiplicative_final_component)
    p7 = (x + y, 1, additive_final_component)
    p8 = (x + y, 1, multiplicative_final_component)
    p9 = (x + y, 1, additive_final_component)
    p10 = (x, y, multiplicative_final_component)

    params_list = [p1, p2, p4, p5, p6, p7, p8, p9, p10]

    templates = [
        # T0: (X+Y) * (1+Z)
        "A team starts with {X} members. They recruit {Y} new members. Then each current member recruits {Z} additional people. How many people are there now on the team? Give the answer only and nothing else.",
        # T1: (X+Y) * Z
        "A chef has {X} apples and buys {Y} more. He then decides to multiply his total stock by {Z} for a large banquet. How many apples does he have now? Give the answer only and nothing else.",
        # T2: (X+Y) + Z
        "A bus has {X} passengers. At the stop, {Y} more get on. Later, {Z} more people get on the bus. How many passengers are there now? Give the answer only and nothing else.",
        # T3: (X-Y) * Z
        # "A store had {X} laptops, but sold {Y} of them. The manager then ordered {Z} times the remaining amount to restock. How many laptops are there now? Give the answer only and nothing else.",
        # T4: (X-Y) + Z
        "A gamer starts with {X} points but loses {Y} points in a match. In the bonus round, they gain {Z} additional points. How many points do they have now? Give the answer only and nothing else.",
        # T5: (X*Y) * Z
        "A warehouse has {X} stacks of {Y} boxes each. The manager decides to multiply the total count by {Z} for the inventory report. How many boxes are recorded? Give the answer only and nothing else.",
        # T6: (X*Y) + Z
        "A gardener plants {X} rows of {Y} trees. Later, they plant {Z} more trees. How many trees are there in total? Give the answer only and nothing else.",
        # T7: (X/Y) * Z
        "A lottery prize of {X} dollars is split between {Y} winners. Each winner invests their share to make it grow {Z} times larger. What is the final amount each winner has? Give the answer only and nothing else.",
        # T8: (X/Y) + Z
        "A construction site has {X} tons of sand divided into {Y} piles. A truck dumps {Z} more tons of sand onto one pile. How many tons of sand are in that pile now? Give the answer only and nothing else.",
        # T9: (X+Y) * Z
        "A library starts with {X} books. They receive a donation of {Y} books. The librarian then calculates that they need {Z} times this total to fill the shelves. What is the target number of books? Give the answer only and nothing else.",
    ]

    result_prompts = []
    for i, tmpl in enumerate(templates):
        cur_x, cur_y, cur_z = params_list[i]
        result_prompts.append(tmpl.format(X=cur_x, Y=cur_y, Z=cur_z))

    return result_prompts


# ==========================================
# 2. The Verification Logic
# ==========================================


def extract_nums(text):
    """Finds all integers in the prompt string."""
    return [int(n) for n in re.findall(r"\d+", text)]


def test_prompts():
    print("Starting verification for X, Y, Z in range [1, 10]...")

    # Define logic verifiers for each template index (0-9)
    # Each lambda takes the extracted numbers [p0, p1, p2] and returns (Intermediate, Final)
    verifiers = {
        0: lambda p: (
            p[0] + p[1],
            (p[0] + p[1]) * (1 + p[2]),
        ),  # (X+Y) then each recruits Z -> (X+Y)(1+Z)
        1: lambda p: (p[0] + p[1], (p[0] + p[1]) * p[2]),  # (X+Y) * Z
        2: lambda p: (p[0] + p[1], (p[0] + p[1]) + p[2]),  # (X+Y) + Z
        3: lambda p: (p[0] - p[1], (p[0] - p[1]) * p[2]),  # (X-Y) * Z
        4: lambda p: (p[0] - p[1], (p[0] - p[1]) + p[2]),  # (X-Y) + Z
        5: lambda p: (p[0] * p[1], (p[0] * p[1]) * p[2]),  # (X*Y) * Z
        6: lambda p: (p[0] * p[1], (p[0] * p[1]) + p[2]),  # (X*Y) + Z
        7: lambda p: (
            p[0] // p[1],
            (p[0] // p[1]) * p[2],
        ),  # (X/Y) * Z (using integer division)
        8: lambda p: (
            p[0] // p[1],
            (p[0] // p[1]) + p[2],
        ),  # (X/Y) + Z (using integer division)
        9: lambda p: (p[0] + p[1], (p[0] + p[1]) * p[2]),  # (X+Y) * Z
    }

    count = 0
    errors = 0

    for x in range(1, 11):
        for y in range(1, 11):
            for z in range(1, 11):
                # These are the "Truth" values we want every template to reach
                expected_intermediate = x + y
                expected_final = (x + y) * (1 + z)

                # Generate the 10 prompts
                prompts = generate_consistent_prompts(x, y, z)

                for i, prompt in enumerate(prompts):
                    # Extract the numbers actually present in the string
                    nums = extract_nums(prompt)

                    # Ensure we found exactly 3 numbers
                    if len(nums) != 3:
                        print(
                            f"Format Error at {x},{y},{z} Template {i}: Found {len(nums)} numbers {nums}"
                        )
                        errors += 1
                        continue

                    # Calculate results based on the text of the prompt
                    calc_inter, calc_final = verifiers[i](nums)

                    # Verify Intermediate
                    if calc_inter != expected_intermediate:
                        print(f"FAIL Intermediate | Base({x},{y},{z}) | Tmpl {i}")
                        print(f"Prompt: {prompt}")
                        print(f"Expected: {expected_intermediate}, Got: {calc_inter}")
                        errors += 1

                    # Verify Final
                    if calc_final != expected_final:
                        print(f"FAIL Final | Base({x},{y},{z}) | Tmpl {i}")
                        print(f"Prompt: {prompt}")
                        print(f"Expected: {expected_final}, Got: {calc_final}")
                        errors += 1

                    count += 1

    if errors == 0:
        print(
            f"SUCCESS: Tested {count} combinations. All intermediate and final answers matched."
        )
    else:
        print(f"FINISHED with {errors} errors.")


if __name__ == "__main__":
    test_prompts()
