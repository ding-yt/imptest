#!/usr/bin/perl

my $input = $ARGV[0];
my $output = $ARGV[1];

sub write_seq_per_sample(){
	my $seq = $_[0];
	my $header = $_[1];
	$header =~ /(.+\/)/;
	my $dir = $1;
	
	if (! -e $dir){
		print "creating $dir\n";
		mkdir $dir;
	}
	
	foreach $k (keys %{$seq}){
		my $file1 = $header.$k.'_hap1.txt';
		my $file2 = $header.$k.'_hap2.txt';
		print ("$file1\n");
		if (-e $file1){
			open(F, ">>$file1");
			select F;
			print($seq->{$k}->[0]);
			close F;
		}else{
			open(F, ">$file1");
			select F;
			print($seq->{$k}->[0]);
			close F;
		}
		
		if (-e $file2){
			open(F, ">>$file2");
			select F;
			print($seq->{$k}->[1]);
			close F;
		}else{
			open(F, ">$file2");
			select F;
			print($seq->{$k}->[1]);
			close F;
		}
		
	}
	select STDOUT;

}


my @samples;
my $seq;
my $count = 0;
my $diff_length_SNP_count = 0;

open (FILE, "$input") or die "can't open $input\n";
while(<FILE>){
	if (/^##/){
		next;
	}elsif(/^#/){
		my @temp = split /\s+/, $_; # 0-8 are infos, samples begin at 9
		for (my $i=9; $i<=$#temp;$i++){
			push @samples, $temp[$i];
		}
		print ("total sample: $#samples+1\n");
	}elsif(/VT=SNP/){
		$count ++;
		if ($count % 10000 ==0){
			#&write_seq_per_sample($seq,$output);
			foreach (@samples){
				my $l1 = length($seq->{$_}->[0]);
				my $l2 = length($seq->{$_}->[1]); 
				print "$_:$l1 $l2\t";
				$seq->{$_}->[0] = ();
				$seq->{$_}->[1] = ();
			}
			print "\n";
		}
		my $line = $_;
		my @temp = split /\s+/, $line; 
		my @allel = split /,/, $temp[4]; #alt
		unshift @allel, $temp[3]; # ref is $temp[3]
		
		my $lengthdiff = 0;
		my $max_length = length($temp[3]);
		for (my $i=0; $i<=$#allel;$i++){
			if(length($allel[$i]) != length($temp[3])){
				print "$temp[0] $temp[1]: $temp[3]\t$temp[4]\n";
				$lengthdiff = 1;
			}
			if (length($allel[$i]) > $max_length){
				$max_length = length($allel[$i]);
			}
		
		}
		$diff_length_SNP_count += $lengthdiff;
		
		if ($lengthdiff == 1){
			print "max length: $max_length\n";
			for (my $i=0; $i<=$#allel;$i++){
				my $add = $max_length - length($allel[$i]);
				#print "add $add ";
				for (my $j=0; $j<$add;$j++){
					$allel[$i] .= "-";
				}
			}
			#print "\n";
		
		}
		
		for (my $i=1; $i<=$#allel;$i++){
			if(length($allel[$i]) != length($allel[0])){
				print "Padded $temp[0] $temp[1]: $allel[0]\t$allel[$i]\n";
			}
			
		}
		
		for (my $i=9; $i<=$#temp;$i++){
			my @hap = split /\|/, $temp[$i];
			$seq->{$samples[$i-9]}->[0] .= $allel[$hap[0]];
			$seq->{$samples[$i-9]}->[1] .= $allel[$hap[1]];
		}
	
	}

}
close FILE;

#&write_seq_per_sample($seq,$output);
#print ("$seq->{$samples[$i-9]}->[0] \n");
select STDOUT;
print ("total SNP $count\n");
print ("SNP with different length $diff_length_SNP_count\n")
