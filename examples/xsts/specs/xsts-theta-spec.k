requires "xsts-verification.k"

module XSTS-THETA-SPEC
    imports XSTS-VERIFICATION

    claim [trivial]: 
        <k> 
            var $a : integer = 0 
            trans { .Sequence } 
            init { .Sequence } 
            env { .Sequence } 
            prop { true } 
            => 
            .K 
        </k>
        <property> .Prop => ?_PROP_END </property>
        <enum-definitions> .Map => ?_ENUM_END </enum-definitions>
        <variable-types> _VARS => ?_VARS_END </variable-types>
        <transition> .K => ?_TRANS_END </transition>
        <init> .K => ?_INIT_END </init>
        <env> .K => ?_ENV_END </env>
        <variable-values> _VALS => ?_VALS_END </variable-values>
        <linking-declarations> .K => ?_DECL_END </linking-declarations>
        <evaluator> .K => ?_RUNNER_END </evaluator>
        <status> START => ?_STATUS_END </status>
        <return-status> NO-RETURN => ?_STATUS_RETURN </return-status>

    claim [example]: 
        <k> 
            var $x: integer = 0
            var $col: integer = 0
            var $b: boolean
            
            trans {
                havoc $b;
                choice {
                    assume $x < 5;
                    $col := 1;
                } or {
                    $x := $x - 1;
                }
            }

            init { .Sequence }
            
            env {
                $x := $x + 1;
            }

            prop { $col != 1 }
            => 
            .K 
        </k>
        <property> .Prop => ?_PROP_END </property>
        <enum-definitions> .Map => ?_ENUM_END </enum-definitions>
        <variable-types> _VARS => ?_VARS_END </variable-types>
        <transition> .K => ?_TRANS_END </transition>
        <init> .K => ?_INIT_END </init>
        <env> .K => ?_ENV_END </env>
        <variable-values> _VALS => ?_VALS_END </variable-values>
        <linking-declarations> .K => ?_DECL_END </linking-declarations>
        <evaluator> .K => ?_RUNNER_END </evaluator>
        <status> START => ?_STATUS_END </status>
        <return-status> NO-RETURN => ?_STATUS_RETURN </return-status>

endmodule