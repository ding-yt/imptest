#!/usr/bin/perl
use List::Util 'shuffle';
use File::Copy;

my $dir = "/work/yd44/imputation/sample_perl/chr1";
my $subpop_file = "/dscrhome/yd44/imputation/20130502.phase3.sequence.index";
my @chr = (1);
#my @chr = (1,10,11,2,20,21,22,3,4,5,6,7,8,9);
my $devdir = "/work/yd44/imputation/sample_perl/dev";
my $testdir = "/work/yd44/imputation/sample_perl/test";


my @all_samples;
my %subpop;
my %count;

opendir (DIR, "$dir") or die "can't open $dir\n";
my @files = readdir DIR;
closedir DIR;
print "total file number ".@files.".\n";
print "$files[10]\n";

foreach my $f (@files){
	if ($f =~ /_(.+)_hap1/){
		#$f =~ m/_(.+)_hap1/;
		push @all_samples, $1;
	}
}
print "total sample number ".@all_samples.".\n";

open ($fh, "$subpop_file") or die "can't open $subpop_file\n";
my $header = <$fh>;
while (<$fh>){
	my @temp = split /\t/, $_;
	$subpop{$temp[9]} = $temp[10];
	#print "$temp[9]\t $temp[10]\n";
}
close $fh;


my @shuffled = shuffle(@all_samples);



for (my $c = 0; $c <= $#chr;$c++){
	#prepare dev set
	my $dev_chrDir = $devdir."/chr".$chr[$c];
	#print "$dev_chrDir\n";
	if (! -e $dev_chrDir){
		mkdir $dev_chrDir;
	}
	for (my $i=2000;$i<=2249;$i++){
		my $datafile1 = "/work/yd44/imputation/sample_perl/chr".$chr[$c]."/chr".$chr[$c]."_".$all_samples[$i]."_";
		#my $datafile2 = "/work/yd44/imputation/sample_perl/chr".$chr[$c]."/chr".$chr[$c]."_".$all_samples[$i]."_hap2.txt";
		my $cmd = "mv $datafile1".'* '.$dev_chrDir;
		print "$cmd\n";
		#system($cmd);
	}
	
	my $test_chrDir = $testdir."/chr".$chr[$c];
	if (! -e $test_chrDir){
		mkdir $test_chrDir;
	}
	for (my $i=2250;$i<=2503;$i++){
		my $datafile1 = "/work/yd44/imputation/sample_perl/chr".$chr[$c]."/chr".$chr[$c]."_".$all_samples[$i]."_";
		#my $datafile2 = "/work/yd44/imputation/sample_perl/chr".$chr[$c]."/chr".$chr[$c]."_".$all_samples[$i]."_hap2.txt";
		my $cmd = "mv $datafile1".'* '.$test_chrDir;
		print "$cmd\n";
		#system($cmd);
	}
}

my $count = &count_subpop(\%subpop,\@all_samples,0,1999,"/dscrhome/yd44/imputation/train.stat");

print "In training set:\n";
my $sum = 0;
foreach my $k (keys %{$count}){
	$sum += $count->{$k};
	print "$k\t$count->{$k}\n";
}
print "total: $sum\n";


$count = &count_subpop(\%subpop,\@all_samples,2000,2249,"/dscrhome/yd44/imputation/dev.stat");
print "\nIn dev set:\n";
$sum = 0;
foreach my $k (keys %{$count}){
 	$sum += $count->{$k};
	print "$k\t$count->{$k}\n";
}
print "total: $sum\n";

$count = &count_subpop(\%subpop,\@all_samples,2250,2503,"/dscrhome/yd44/imputation/test.stat");
print "\nIn test set:\n";
$sum = 0;
foreach my $k (keys %{$count}){
 	$sum += $count->{$k};
	print "$k\t$count->{$k}\n";
}
print "total: $sum\n";



sub count_subpop{
	my ($subpop,$list,$start,$end, $file) = @_;
	my $count;
	my $sum = 0;
	for (my $i=$start;$i<=$end;$i++){
	if (exists $subpop->{$list->[$i]}){
		$count->{$subpop->{$list->[$i]}} ++;
	}else{
		print "can't find subpop info for $list->[$i]\n"; 
	}
	}
	
	open ($fh, ">$file");
	select $fh;
	for (my $i=$start;$i<=$end;$i++){
		print "$list->[$i]\t";
	}
	print "\n";
	foreach my $k (keys %{$count}){
 		$sum += $count->{$k};
		print "$k\t$count->{$k}\n";
	}
	print "total: $sum\n";
	close $file;
	select STDOUT;
	print "$file ready\n";
	
	return $count;
}


