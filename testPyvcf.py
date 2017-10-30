#!/usr/bin/env python3

import vcf
import re
import argparse
import os
from pathlib import Path


def write_seq_per_sample(seq,header): #header is the path/chr?/chr?_
	dir = re.search("(.+\/)",header).group(1)
	print("result in dir: " + dir)
	if not os.path.isdir(dir):
		print("creating dir: "+dir)
		os.mkdir(dir)
	for key in seq.keys():
		file1 = header + key + '_hap1.txt'
		file2 = header + key + '_hap2.txt'
		if Path(file1).is_file():
			f = open(file1,'a')
			f.write(seq[key][0])
			f.close()
		else:
			f = open(file1,'w')
			f.write(seq[key][0])
			f.close()
		
		if Path(file2).is_file():
			f = open(file2,'a')
			f.write(seq[key][1])
			f.close()
		else:
			f = open(file2,'w')
			f.write(seq[key][1])
			f.close()	




parser = argparse.ArgumentParser()
parser.add_argument("vcf", help="path to the vcf file that needs processing")
parser.add_argument("out", help="output file")

args = parser.parse_args()

count_SNP = 0
seq = {}
progess = 0


vcf_reader = vcf.Reader(open(args.vcf,"r"))
record = next(vcf_reader)
for sample in record.samples:
	t_list = ["",""]
	seq[sample.sample] = t_list

vcf_reader = vcf.Reader(open(args.vcf,"r"))
for record in vcf_reader:
#	if any("SNP" in s for s in record.INFO["VT"]):
	progess += 1
	if progess % 5000 ==0:
		print(str(progess)+' lines processed')
		write_seq_per_sample(seq,args.out)
		for sample in record.samples:
			seq[sample.sample] = ["",""]

		
	if re.match("SNP",  record.INFO["VT"][0]):
		count_SNP += 1
		for sample in record.samples:
			g1,g2 = sample['GT'].split("|")
			if g1 == 0:
				g1 = record.REF
			else:
				g1 = record.ALT[int(g2)-1]
			if g2 == 0:
				g2 = record.REF
			else:
				g2 = record.ALT[int(g2)-1]
			seq[sample.sample][0] += str(g1)
			seq[sample.sample][1] += str(g2)
	 		

write_seq_per_sample(seq,args.out)		
print("total SNP: "+str(count_SNP))
print("total samples: ",len(seq))

# f = open(args.out,'w')
# for key in seq.keys():
# 	f.write(key+':'+seq[key][0]+'\n')
# 	f.write(key+':'+seq[key][1]+'\n')
# f.close()

# 	print(record.CHROM)
# 	print(record.POS)

#record = next(vcf_reader)
#print(type(record.INFO["VT"][0]))

#print(record.samples)

# for sample in record.samples:
# 	print(sample['GT'])

#all_keys = set().union(*(d.keys() for d in record.samples))

# print( "###################")
# print(record.samples[0]["GT"])
# print(record.ALT)
# print(type(record.samples[0]))
##print(record.genotype(record.samples.keys()[0])['GT'])