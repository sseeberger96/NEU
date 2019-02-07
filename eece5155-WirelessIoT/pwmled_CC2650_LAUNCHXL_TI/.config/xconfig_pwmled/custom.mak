## THIS IS A GENERATED FILE -- DO NOT EDIT
.configuro: .libraries,em3 linker.cmd package/cfg/pwmled_pem3.oem3

# To simplify configuro usage in makefiles:
#     o create a generic linker command file name 
#     o set modification times of compiler.opt* files to be greater than
#       or equal to the generated config header
#
linker.cmd: package/cfg/pwmled_pem3.xdl
	$(SED) 's"^\"\(package/cfg/pwmled_pem3cfg.cmd\)\"$""\"/Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI/.config/xconfig_pwmled/\1\""' package/cfg/pwmled_pem3.xdl > $@
	-$(SETDATE) -r:max package/cfg/pwmled_pem3.h compiler.opt compiler.opt.defs
