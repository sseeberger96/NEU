# invoke SourceDir generated makefile for pwmled.pem3
pwmled.pem3: .libraries,pwmled.pem3
.libraries,pwmled.pem3: package/cfg/pwmled_pem3.xdl
	$(MAKE) -f /Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI/src/makefile.libs

clean::
	$(MAKE) -f /Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI/src/makefile.libs clean

