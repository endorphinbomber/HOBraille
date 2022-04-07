

txtf = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/_GITHUB/HOBraille/vollschrift.tab'
txto = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/_GITHUB/HOBraille/vollschrift_converted.tab'

with open(txtf) as fp:
   lines = fp.readlines()
   
cleanl = []
for line in lines:
    cleanl.append(  line.rsplit(' [', 1)[0].strip()  )


newl = []
for line in cleanl:
    left = '\''+line.rsplit('=')[0]+'\''
    right = bytes.fromhex(line.rsplit('=')[1][1:]).decode('utf-8')[1:]
    newl.append(left+': '+'\''+right+'\'')
    
with open(txto, 'w') as f:
    for item in newl:
        f.write("%s\n" % item)
