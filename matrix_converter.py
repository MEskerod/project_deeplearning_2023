from io import TextIOWrapper
import matplotlib.pyplot as plt
import numpy as np
import argparse, os


def read_bpseq(file: TextIOWrapper) -> tuple():
    """
    """

    sequence = ""
    pairs = []
    
    lines = [line.split() for line in file.readlines()]

    #Remove header - if any
    header_lines = 0
    for line in lines: 
        if line[0] == '1': 
                break
        else: 
            header_lines += 1

    lines = lines[header_lines:]

    #Make sequence in bp list
    for line in lines: 
        sequence += line[1]
        if line[2] != '0': 
            pairs.append((int(line[0])-1, int(line[2])-1)) #The files start indexing from 1
    return sequence, pairs

def read_ct(file: TextIOWrapper) -> tuple():
    """
    """
    sequence = ""
    pairs = []

    lines = [line.split() for line in file.readlines()]

    #Remove header - if any
    header_lines = 0
    for line in lines: 
        if line[0] == '1': 
                break
        else: 
            header_lines += 1

    lines = lines[header_lines:]
    
    for line in lines: 
        sequence += line[1]
        if line[4] != '0': 
            pairs.append((int(line[0])-1, int(line[4])-1)) #The files start indexing from 1

    return sequence, pairs

def make_matrix_from_sequence(sequence: str) -> np.array:
    """
    """
    colors = {"not_paired": [255, 255, 255],
              "unpaired": [64, 64, 64],
              "GC": [0, 255, 0],
              "CG": [0, 128, 0],
              "UG": [0, 0, 255],
              "GU": [0, 0, 128],
              "UA": [255, 0, 0],
              "AU": [128, 0, 0]}
    basepairs = ["GC", "CG", "UG", "GU", "UA", "AU"]

    N = len(sequence)
    
    matrix = np.full((N,N,3),255, dtype="uint8")

    for i in range(N):
        for j in range(N):
            pair = sequence[i] + sequence[j]
            if i == j: 
                matrix[i, j, :] = colors["unpaired"]
            elif pair in basepairs:
                matrix[i, j, :] = colors[pair]
    
    return matrix


def make_matrix_from_basepairs(sequence: str, pairs: list) -> np.array: 
    """
    """
    black = [0, 0, 0]
    
    N = len(sequence)
    matrix = np.full((N,N,3),255, dtype="uint8")

    for pair in pairs: 
        matrix[pair[0], pair[1], :] = black

    return matrix


def save_matrix(matrix: np.array, name: str) -> None: 
    plt.imsave(name, matrix)

def main(): 
    argparser = argparse.ArgumentParser(prog = "MatrixConverter",
                                        description = "Converts .bpseq or .ct files into matrices representing the possible base pairs and the structure.\n\
                                            If input file is .bpseq use -b/--bpseq.\nIf input file is .ct use -c/--ct.")
    argparser.add_argument("-b", "--bpseq", type=argparse.FileType('r'), help=".bpseq file")
    argparser.add_argument("-c", "--ct", type=argparse.FileType('r'), help=".ct file")
    argparser.add_argument("-o", "--output_dir", help = "Output directory. If not supplied the files are outputted to the current directory")

    args = argparser.parse_args()

    if args.bpseq and args.ct:
        raise ValueError("Only one input file is allowed")
    
    elif args.bpseq: 
        sequence, pairs = read_bpseq(args.bpseq)
        file_name = os.path.splitext(os.path.basename(args.bpseq.name))[0]
    
    elif args.ct: 
        sequence, pairs = read_ct(args.ct)
        file_name = os.path.splitext(os.path.basename(args.ct.name))[0]
    
    bp_matrix = make_matrix_from_sequence(sequence)
    structure_matrix = make_matrix_from_basepairs(sequence, pairs)

    if args.output_dir: 
        save_matrix(bp_matrix, os.path.join(args.output_dir, file_name + "_bp.png"))
        save_matrix(structure_matrix, os.path.join(args.output_dir, file_name + "_struct.png"))
    else: 
        save_matrix(bp_matrix, file_name + "_bp.png")
        save_matrix(structure_matrix, file_name + "_struct.png")
        

if __name__ == "__main__": 
    main()








