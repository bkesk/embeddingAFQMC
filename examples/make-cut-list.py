fname='count-vs-R.occ.dat'
Ntotal = 78

fout='local-cuts-'+fname

entries=[]

f=open(fname,'r')
for line in f:
    entries.append(line.split())
f.close()

entries.sort(key=lambda x : int(x[1]))

trans = lambda x : Ntotal - int(x)

# before output, we want to cut away values
trimmed = [entries[0]]
current = trans(entries[0][1])
for entry in entries:
    if  trans(entry[1]) < current:
        trimmed.append(entry)
        current =trans(entry[1])

fo=open(fout,'w')
for entry in trimmed:
    print(entry)
    fo.write(f'{trans(entry[1])} {float(entry[0]):.2f}\n')
fo.close()
