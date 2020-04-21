# (t)csh-specific configuration script.
#
# Usage: 
#
#  source source_me.csh
#

if ( `basename -- $0` == "source_me.csh" ) then
    echo "Error: this script should not be executed, only sourced."
    exit 1
endif

#source `dirname -- $0`/lib/core/site/detect_site.csh
#source `dirname -- $0`/lib/core/site/$site/source_me.csh

setenv path1 $PWD/utils/
setenv PYTHONPATH ${path1}
rehash

if ( -t 1 ) then
    echo "$site site settings activated."
endif

