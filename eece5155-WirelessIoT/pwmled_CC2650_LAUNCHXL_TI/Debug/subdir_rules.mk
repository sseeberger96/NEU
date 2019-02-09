################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Each subdirectory must supply rules for building sources it contributes
%.obj: ../%.c $(GEN_OPTS) | $(GEN_FILES)
	@echo 'Building file: "$<"'
	@echo 'Invoking: ARM Compiler'
	"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.4.LTS/bin/armcl" -mv7M3 --code_state=16 --float_support=vfplib -me --include_path="/Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI" --include_path="/Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI" --include_path="/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/cc26xxware_2_24_03_17272" --include_path="/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.4.LTS/include" --define=ccs -g --diag_warning=225 --diag_warning=255 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi --preproc_with_compile --preproc_dependency="$(basename $(<F)).d_raw" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

build-2136563087:
	@$(MAKE) --no-print-directory -Onone -f subdir_rules.mk build-2136563087-inproc

build-2136563087-inproc: ../pwmled.cfg
	@echo 'Building file: "$<"'
	@echo 'Invoking: XDCtools'
	"/Applications/ti/xdctools_3_32_02_25_core/xs" --xdcpath="/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/tidrivers_cc13xx_cc26xx_2_21_00_04/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/bios_6_46_01_37/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/uia_2_01_00_01/packages;" xdc.tools.configuro -o configPkg -t ti.targets.arm.elf.M3 -p ti.platforms.simplelink:CC2650F128 -r release -c "/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.4.LTS" --compileOptions "-mv7M3 --code_state=16 --float_support=vfplib -me --include_path=\"/Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI\" --include_path=\"/Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI\" --include_path=\"/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/cc26xxware_2_24_03_17272\" --include_path=\"/Applications/ti/ccsv8/tools/compiler/ti-cgt-arm_18.1.4.LTS/include\" --define=ccs -g --diag_warning=225 --diag_warning=255 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi  " "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

configPkg/linker.cmd: build-2136563087 ../pwmled.cfg
configPkg/compiler.opt: build-2136563087
configPkg/: build-2136563087


