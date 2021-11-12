import os


def get_files_list(filetype=".txt"):
    """ Get all the files in the current directory with the
        specified filetype.
    """
    out = []
    for filename in os.listdir(os.getcwd()):
        if filename.endswith(filetype):
            out.append(filename)
    return out

def pretty_print_story(story_string):
    """ Print the story so that when the lines are over 50 characters
        it goes to the next line.
    """
    character_max_limit = 50
    story_words = story_string.split(" ")
    current_line = ""
    for word in story_words:
        current_line += word + " "
        if len(current_line)>character_max_limit:
            # the line is ready to print
            print(current_line)
            # the line is cleared for the next words
            current_line = ""
    # finally print whatever is left when done
    print(current_line)


# This shows the user the available stories
possible_filenames_list = get_files_list()
print("Possible madlib files:")
for filename in possible_filenames_list:
    print("\t"+filename)

# Here I get the use to choose the file
user_txt_selection = input("What is the name of the Madlib file? ")
# user_txt_selection = "madlib_story_1.txt"
# If the use doesnt put the txt, then add it
if user_txt_selection[-4:] != ".txt":
    user_txt_selection += ".txt"

# Open the 
raw_txt_data = [line.strip() for line in open(user_txt_selection, 'r')]

# Make a empty string to fill with the madlib story
output_string = ""

# Now step through each line by the index value
for i in range(len(raw_txt_data)):
    # If the index value is even, it is a story line
    # else the index value is odd, it is a input line
    if i%2==0:  # the remainder is 0 is it is even
        # if the story line has a period, then there is no space
        if raw_txt_data[i][0] == ".":
            output_string += raw_txt_data[i]
        else:
            output_string += " " + raw_txt_data[i]
    else:
        user_input_string = input(raw_txt_data[i]+": ")
        output_string += " " + user_input_string

pretty_print_story(output_string)
