#!/usr/bin/env python3
import os
import re
from pathlib import Path

shell_dir = '/dscrhome/yd44/imputation/shell'
log_dir = '/work/yd44/imputation/log'
gz_dir = '/work/yd44/imputation'
preprocess_script = '/dscrhome/yd44/imputation/testPyvcf.py'
preprocess_script_perl = '/dscrhome/yd44/imputation/getSeq.pl'

#chrom_list = [1,2,3,4,5,6,7,8,9,11,20,21,22]
#chrom_list = [10]
chrom_list = [1,2,3,4,5,6,7,8,9,10,11,20,21,22]

# for filename in os.listdir(gz_dir):
# 	if re.search( "gz$",filename):
# 		gz_file.append(filename)
# 
# for file in gz_file:
# 	chrom = re.search('(chr\d+)\.',file,re.IGNORECASE).group(1)
# 	print(chrom)
# 	vcffile = gz_dir +'/'+ file
# 	vcffile = re.sub('\.gz','',vcffile)
# 	vcffile = Path(vcffile)
# 	print(vcffile.is_file())


for chrom in chrom_list:
	gz_file = gz_dir +'/'+ 'ALL.chr' + str(chrom) +'.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'
	vcf_file = gz_dir +'/'+ 'ALL.chr' + str(chrom) +'.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf'
	
	shell_file = shell_dir +'/pre_chr'+ str(chrom) + '.sh'
	log_file = log_dir +'/pre_chr'+ str(chrom) + '.log'
	error_file = log_dir +'/pre_chr'+ str(chrom) + '.error'
	
	#output_file = gz_dir +'/sample/chr'+ str(chrom) + '/chr' + str(chrom)+'_'
	output_file = gz_dir +'/sample_perl/chr'+ str(chrom) + '/chr' + str(chrom)+'_'
	
	s = open(shell_file,'w')
	s.write('#!/bin/sh\n')
	s.write('#SBATCH --output='+log_file+'\n')
	s.write('#SBATCH --job-name=prchr'+str(chrom)+'\n')
	s.write('#SBATCH --error='+error_file+'\n')
	s.write('#SBATCH --mem=5G\n\n') 
	s.write('source /opt/apps/sdg/sdg_bashrc\n')
	s.write('source /dscrhome/yd44/.bashrc\n')
	s.write('source /dscrhome/yd44/software/virtualpy/python3.4/bin/activate\n\n')
	
	if Path(vcf_file).is_file():
		print('### vcf for chr'+str(chrom)+' found')
		s.write('### vcf for chr'+str(chrom)+' found\n')
	else:
		print('### need to generate vcf for chr'+str(chrom)+'')
		s.write('### need to generate vcf for chr'+str(chrom)+'\n')
		cmd = 'gunzip -c ' + gz_file + ' > ' + vcf_file
		s.write(cmd+'\n')
	
	#cmd = 'python ' + preprocess_script +' '+ vcf_file +' '+ output_file
	cmd = 'perl ' + preprocess_script_perl +' '+ vcf_file +' '+ output_file
		
	s.write(cmd+'\n')
	s.close()
	
	print('sbatch ' + shell_file + '\n')
		
	
	