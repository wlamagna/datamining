#!/usr/bin/perl
# 28-Jan-2018 - genera un resumen de los datos obtenidos de el banco mundial de desarrollo
# 		con el fin de generar un resumen de los datos abiertos.
#
#

my $filename = @ARGV[0];

open A,"$filename" or die ("no pude abrir $filename");

my %summary = ();

# Nombre del Proyecto     País    No. de identificación del proyecto      Monto del Compromiso    Estatus Fecha de aprobación
while (<A>) {
	chomp;
	next if (/^Nombre del/);
	my ($proyecto, $pais, $id_proj, $monto, $status, $fecha) = split(/\t/);
	my ($year, $month, $day) = $fecha =~ /([0-9]{4})-([0-9]{2})-([0-9]{2}).*/g;
	if (!(defined($summary{$year}))) {
		$summary{$year} = $monto;
		open B, ">deuda/bm_$year.html";
		print B "<html><head>
<meta charset=\"UTF-8\">
<style type=\"text/css\">
body {
font: 10px sans-serif;
}
</style>";
		print B "<b>$fecha</b>&nbsp;<font color=\"red\">$monto</font>&nbsp;$proyecto<br />\n";
		close B;
	} else {
		$summary{$year} += $monto;
		open B, ">>deuda/bm_$year.html";
		print B "<b>$fecha</b>&nbsp;<font color=\"red\">$monto</font>&nbsp;$proyecto<br />\n";
		close B;
	}
}

close A;
print "cantidad\tdate\n";
foreach my $y (sort keys %summary) {
	print "$summary{$y}\t$y" . "0101\n";
}