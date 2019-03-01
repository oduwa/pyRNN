stripped = ""
with open('jaden.txt', 'r') as f:
    read_data = f.read()
    read_data = str.replace(read_data, '\n', ' \\n ')
    stripped = "".join(c for c in read_data if c not in ('!','.',':',',','\"','-', '\''))

with open('jaden_stripped.txt', 'w') as f:
    f.write(stripped)
