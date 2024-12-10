# if this is not automatically loaded by gdb, look into warnings 
#
# One fix was to disable safe path loading
# echo "add-auto-load-safe-path /" >>  $HOME/.config/gdb/gdbinit
#
#
# 1. enable debug info  https://ubuntu.com/server/docs/service-debuginfod
# echo "set debuginfod enabled on" >>  $HOME/.gdbinit

## NOTE: gdb might not load this by default due to security reasons
##       watch out the warning messages when you start gdb for resolving this
##       one fix is to add the following line to your gdbinit file
## For me, I had to add a line to "$HOME/.config/gdb/gdbinit"


# sample user-defined command
define hello
p "hello world" 
end


define ptensor
  call std::cout << $arg0
  call fflush(0)
end

