# if this is not automatically loaded by gdb, look into warnings 
#
# One fix was to disable safe path loading
# echo "add-auto-load-safe-path /" >>  $HOME/.config/gdb/gdbinit
#
#
# 1. enable debug info  https://ubuntu.com/server/docs/service-debuginfod
# echo "set debuginfod enabled on" >>  $HOME/.gdbinit


# sample user-defined command
define hello
p "hello world" 
end

#source tools/gdb/pytorch-gdb.py

#b /mnt/home/tg/work/repos/tahoma/src/model/transformer_nmt.hpp:59
