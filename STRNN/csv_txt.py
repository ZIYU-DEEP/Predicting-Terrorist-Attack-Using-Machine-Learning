import os

if __name__ == "__main__":

    with open('gtd.csv', 'r') as input_file:
        lines = [line.split(",") for line in input_file.readlines()]

    count = 0
    for line in lines:
        if line[-1] in ("", "\n", "999999\n"):
            count += 1
    print("Number of rows with missing location_id: ", count)

    if not os.path.isfile("gtd.txt"):
        os.remove("gtd.txt")
    with open ('gtd.txt', 'a') as output_file:
        for line in lines[1:]:
            if any([line[0] in ("9999", ""), line[-1] in ("", "\n", "999999\n"),
                    line[1] == "", line[2] == ""]):
                continue
            output_file.write("\t".join(line))
