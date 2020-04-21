# bash-specific configuration script.
#
# Usage: 
#
#  source source_me.sh
#

if [ `basename -- $0` = "source_me.sh" ]
then
    echo "Error: this script should not be executed, only sourced." 1>& 2
    exit 1
fi

. `dirname -- $0`/lib/core/site/detect_site.sh
. `dirname -- $0`/lib/core/site/$site/source_me.sh
export PYTHONPATH=""
export PYTHONPATH=$PYTHONPATH:$PWD/utils/
echo "$site settings activated."
