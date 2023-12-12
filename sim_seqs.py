import RNA as vrna
import random
import sys
from alive_progress import alive_bar

def generateRNA(number_of_sequences, min, max):
    sequences = []
    for i in range(number_of_sequences):
        length = random.randint(min, max)
        sequence = ""
        for j in range(length):
            sequence += random.choice("ACGU")
        sequences.append(sequence)
    return sequences

def generateRNAwithStructure(sequences):
    structures = []
    with alive_bar(len(sequences)) as bar:
        for seq in sequences:
            fc=vrna.fold_compound(seq)
            (ss, mfe) = fc.mfe()
            structures.append(ss)
            bar()
    return sequences, structures
        
def writeToFile(sequences, structures, filename):
    f = open(filename, "w")
    for i in range(len(sequences)):
        f.write(">" + str(i) + "\n")
        f.write(sequences[i] + "\n")
        f.write(structures[i] + "\n")
    f.close()

def getDotBracketList(file_name):
    '''
    This function takes a file name as input and returns a list of lists
    where each list contains the title, sequence, and dot bracket notation
    of a single RNA sequence.
    '''
    with open(file_name, 'r') as f:
        lines = f.readlines()
    length = len(lines)
    i=0
    db_list = []
    for i in range(length):
        if lines[i].startswith('>'):
            db_list.append([lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()])
    return db_list
    
def dot2ct(dbn):
    try:
        title = dbn[0]
        seq = dbn[1]
        pattern = dbn[2]

        ctstring = []
        stack1 = []
        stack2 = []
        stack3 = []
        stack4 = []
        stack5 = []
        stack6 = []
        stack7 = []
        stack8 = []
        stack9 = []
        stack10 = []
        stack11 = []
        pairs = {}

        for i, c in enumerate(pattern):
            if c == '(':
                stack1.append(i + 1)
            elif c == '[':
                stack2.append(i + 1)
            elif c == '{':
                stack3.append(i + 1)
            elif c == '<':
                stack4.append(i + 1)
            elif c == 'A':
                stack5.append(i + 1)
            elif c == 'B':
                stack6.append(i + 1)
            elif c == 'C':
                stack7.append(i + 1)
            elif c == 'D':
                stack8.append(i + 1)
            elif c == 'E':
                stack9.append(i + 1)
            elif c == 'F':
                stack10.append(i + 1)
            elif c == 'G':
                stack11.append(i + 1)
            elif c == ')':
                pairs[i + 1] = stack1.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == ']':
                pairs[i + 1] = stack2.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == '}':
                pairs[i + 1] = stack3.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == '>':
                pairs[i + 1] = stack4.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == 'a':
                pairs[i + 1] = stack5.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == 'b':
                pairs[i + 1] = stack6.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == 'c':
                pairs[i + 1] = stack7.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == 'd':
                pairs[i + 1] = stack8.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == 'e':
                pairs[i + 1] = stack9.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == 'f':
                pairs[i + 1] = stack10.pop()
                pairs[pairs[i + 1]] = i + 1
            elif c == 'g':
                pairs[i + 1] = stack11.pop()
                pairs[pairs[i + 1]] = i + 1

        for i in range(1, len(pattern) + 1):
            ctstring.append(
                "%d%s%s%s%d%s%d%s%d%s%d" % (i, ' ', seq[i - 1], ' ', i - 1, ' ', i + 1, ' ', pairs.get(i, 0), ' ', i))
        # print(title)
        # print('\n'.join(ctstring))

        with open("vRNA_100_200/"+dbn[0][1:] + '.ct', 'w') as d:
            d.write(title + '\n' + '\n'.join(ctstring))
    except:
        print("Invalid input format")

def main():
    if len(sys.argv) == 5:
        number_of_sequences = int(sys.argv[1])
        min = int(sys.argv[2])
        max = int(sys.argv[3])
        output_file = sys.argv[4]
        sequences = generateRNA(number_of_sequences, min, max)
        sequences, structures = generateRNAwithStructure(sequences)
        writeToFile(sequences, structures, output_file)
    elif len(sys.argv) == 2:
        dbn_list = getDotBracketList(sys.argv[1])
        with alive_bar(len(dbn_list)) as bar:
            for dbn in dbn_list:
                dot2ct(dbn)
                bar()
    else:       
        print("Usage: python sim_seqs.py <number of sequences> <min length> <max length> <output file>")
        print("OR")
        print("Usage: python sim_seqs.py <input file>")
        sys.exit(1)


if __name__ == "__main__":
    main()