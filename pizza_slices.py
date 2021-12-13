
slices_per_pizza = 8

num_people = 5
num_pizzas = 2
num_people_two_slices = 1
print(f"{num_people=}")
print(f"{num_pizzas=}")
print(f"{num_people_two_slices=}")

# people who eat as much as they want
num_people_self_serve = num_people - num_people_two_slices
print(f"{num_people_self_serve=}")

total_slices = num_pizzas*slices_per_pizza
print(f"{total_slices=}")

remaining_slices = total_slices - num_people_two_slices*2
print(f"{remaining_slices=}")

min_slices = remaining_slices // num_people_self_serve
print(f"{min_slices=}")

extra_slices = remaining_slices - min_slices * num_people_self_serve
print(f"{extra_slices=}")

output_str = f"You will divide {total_slices} slices -- "

if min_slices==2:
    output_str += f"{num_people_two_slices+num_people_self_serve-extra_slices} will have 2 slices, "
elif num_people_two_slices>0:
    output_str += f"{num_people_two_slices} will have 2 slices, "

if extra_slices>0:
    output_str += f"{num_people_self_serve-extra_slices} will have {min_slices} slices"
    output_str += f", and {extra_slices} will have {min_slices+1} slices."
else:
    output_str += f"{num_people_self_serve} will have {min_slices} slices."

print(output_str)


