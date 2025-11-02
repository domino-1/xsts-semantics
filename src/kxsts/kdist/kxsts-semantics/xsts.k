module EXPR-SYNTAX
    imports INT
    imports BOOL
    imports ID

    syntax Enum ::= Id "." Id

    syntax Value ::= Int | Bool | Enum
    syntax KResult ::= Value

    syntax Expr ::= Value
                  | Id
                  | ".Expr"
                  | "(" Expr ")" [bracket]
                  > "-" Expr [seqstrict]
                  | "+" Expr [seqstrict]
                  > Expr "*"  Expr [left, seqstrict(1, 2)]
                  | Expr "/"  Expr [left, seqstrict]
                  | Expr "%"  Expr [left, seqstrict]
                  > Expr "+"  Expr [left, seqstrict]
                  | Expr "-"  Expr [left, seqstrict]
                  > Expr ">"  Expr [left, seqstrict]
                  | Expr ">=" Expr [left, seqstrict]
                  | Expr "<"  Expr [left, seqstrict]
                  | Expr "<=" Expr [left, seqstrict]
                  > Expr "==" Expr [left, seqstrict]
                  | Expr "!=" Expr [left, seqstrict]
                  > "!" Expr       [seqstrict]
                  > Expr "&&" Expr [left, seqstrict]
                  > Expr "||" Expr [left, seqstrict]
endmodule

module XSTS-SYNTAX
    imports INT-SYNTAX
    imports BOOL-SYNTAX
    imports ID-SYNTAX
    imports EXPR-SYNTAX

    //syntax Enum ::= Id "." Id a

    syntax Enums ::= List{Id, ","}
    syntax TypeDeclaration ::= "type" Id ":" "{" Enums "}"
    
    syntax TypeName ::= "integer" | "boolean" | Id
    syntax VariableDeclaration ::= "var" Id ":" TypeName
                                 | "var" Id ":" TypeName "=" Expr [strict(3)]

    syntax Assignment ::= Id ":=" Expr ";" [strict(2)]
    syntax Havoc ::= "havoc" Id ";" 
    syntax Assumption ::= "assume" Expr ";" [strict(1)] //Value from Expr, lehet Expr, literalExpression vagy ReferenceExpression

    syntax Operation ::= Choice
                       | Assignment
                       | Havoc
                       | Assumption
    
    syntax Block ::= "{" Sequence "}"
    syntax OrBlocks ::= List{Block, "or"}
    syntax Choice ::= "choice" OrBlocks
    
    syntax Sequence ::= List{Operation, ""}

    syntax Transition ::= "trans" "{" Sequence "}"
    syntax Init ::= "init" "{" Sequence "}"
    syntax Env ::= "env" "{" Sequence "}"
    syntax Prop ::= "prop" "{" Expr "}" | "prop" "{" "}"

    syntax Declaration ::= VariableDeclaration | TypeDeclaration
    syntax Declarations ::= List{Declaration, ""}

    syntax Program ::= Declarations Transition Init Env Prop
                     | Declarations //temp
endmodule

module CONFIGURATION
    imports XSTS-SYNTAX

    syntax Expr ::= ".Prop"
    syntax Status ::= "START" 
                    | "DECLARATIONS"
                    | "INIT"
                    | "TRAN"
                    | "ENV"
                    | "EVAL"
                    | "PROPERTY-EVALUATION-OR-FAIL"
                    | "PROPERTY-SUCCESS"
                    | "PROPERTY-FAIL"
                    | "NO-RETURN"

    syntax Bool ::= isFail( Status ) [function]
                  | isEnumType( TypeName ) [function]

    rule isFail( PROPERTY-FAIL ) => true
    rule isFail( _ ) => false [owise]

    rule isEnumType( integer ) => false
    rule isEnumType( boolean ) => false
    rule isEnumType( _ ) => true [owise]

    configuration <T>
        <k> $PGM:Program </k>
        <xsts>
            <property> .Prop </property>
            <enum-definitions> .Map </enum-definitions>
            <variable-types> .Map </variable-types>
            <transition> .K </transition>
            <init> .K </init>
            <env> .K </env>
        </xsts>
        <variable-values> .Map </variable-values>
        <linking-declarations> .K </linking-declarations>
        <evaluator> .K </evaluator> //transition-eval
        <status> START </status>
        <return-status> NO-RETURN </return-status>
    </T>

    /* status key:
    0: program linking
    1: declaration semantics
    2: init block
    3: tran block
    4: env block
    100: transitions evaluator
    101: property evaluation in progress or evaluated to false
    102: property evaluated

    return status: -1 means no return expected
    */

    /* semantic sorts */
    syntax Assumption ::= "assume" "#active" ";"
    syntax Active ::= "#active" "(" KItem ")"
endmodule

module EXPR-RULES
    imports EXPR-SYNTAX
    imports CONFIGURATION
    imports MAP
    imports STRING

    rule [expr-variable-lookup]: <k> X:Id => VARIABLES [X] ...</k>
         <variable-values> VARIABLES </variable-values> //! requires X in_keys ( VARIABLES )
    
    rule [expr-int-mul]: <k> I1:Int * I2:Int => I1 *Int I2 ...</k>
    rule [expr-int-div]: <k> I1:Int / I2:Int => I1 /Int I2 ...</k>
    rule [expr-int-add]: <k> I1:Int + I2:Int => I1 +Int I2 ...</k>
    rule [expr-int-sub]: <k> I1:Int - I2:Int => I1 -Int I2 ...</k>
    rule [expr-int-grt]: <k> I1:Int > I2:Int => I1 >Int I2 ...</k>
    rule [expr-int-gte]: <k> I1:Int >= I2:Int => I1 >=Int I2 ...</k>
    rule [expr-int-lst]: <k> I1:Int < I2:Int => I1 <Int I2 ...</k>
    rule [expr-int-lte]: <k> I1:Int <= I2:Int => I1 <=Int I2 ...</k>
    rule [expr-int-equ]: <k> I1:Int == I2:Int => I1 ==Int I2 ...</k>
    rule [expr-int-ueq]: <k> I1:Int != I2:Int => I1 =/=Int I2 ...</k>
    rule [expr-int-mod]: <k> I1:Int % I2:Int => I1 modInt I2 ...</k>
    rule [expr-bool-equ]: <k> B1:Bool == B2:Bool => B1 ==Bool B2 ...</k>
    rule [expr-bool-ueq]: <k> B1:Bool != B2:Bool => B1 =/=Bool B2 ...</k>
    rule [expr-bool-or]: <k> B1:Bool || B2:Bool => B1 orBool B2 ...</k>
    rule [expr-bool-and]: <k> B1:Bool && B2:Bool => B1 andBool B2 ...</k>
    rule [expr-enum-equ]: <k> X1:Id.X2:Id == X3:Id.X4:Id => (Id2String(X1) ==String Id2String(X3)) andBool (Id2String(X2) ==String Id2String(X4)) ... </k>
    rule [expr-enum-ueq]: <k> X1:Id.X2:Id != X3:Id.X4:Id => (Id2String(X1) =/=String Id2String(X3)) orBool (Id2String(X2) =/=String Id2String(X4)) ... </k>
    
    rule [expr-int-unary-neg]: <k> - I:Int  => 0 -Int I ... </k>
    rule [expr-int-unary-pos]: <k> + I:Int  => I ... </k>
    rule [expr-bool-negate]: <k> ! B:Bool => notBool B ... </k>
endmodule

module PROGRAM-LINKING
    imports CONFIGURATION
    imports XSTS-SYNTAX
    imports EXPR-RULES
    imports INT

    rule [TEST-simplified-input]: <k> D:Declarations => D:Declarations trans { .Sequence } init { .Sequence } env { .Sequence } prop { true } </k>

    rule [store-input-file-in-config]: <k> D:Declarations trans { T:Sequence } init { I:Sequence } env { E:Sequence } prop { P:Expr } => .K </k>
        <linking-declarations> .K => D </linking-declarations>
        <transition> .K => T </transition>
        <init> .K => I </init>
        <env> .K => E </env>
        <property> .Prop => P </property>
        <status> START => DECLARATIONS </status>
endmodule

module DECLARATIONS
    imports CONFIGURATION
    imports XSTS-SYNTAX
    imports BOOL

    //types
    rule [select-next-type-declaration]: <linking-declarations> T:TypeDeclaration D:Declarations => T ~> D ... </linking-declarations>
         <status> DECLARATIONS </status>

    rule [store-type-declaration]: <linking-declarations> type I:Id : { E:Enums } => .K ... </linking-declarations>
         <enum-definitions> ENUMS => ENUMS [ I <- E ] </enum-definitions>
         <status> DECLARATIONS </status>

    //variables
    rule [select-next-variable-declaration]: <linking-declarations> V:VariableDeclaration D:Declarations => V ~> D ... </linking-declarations>
         <status> DECLARATIONS </status>

    rule [variable-uninit-int]: <linking-declarations> var I:Id : integer => .K ... </linking-declarations>
         <variable-values> VALUES => VALUES [ I <- ?_:Int ] </variable-values>
         <variable-types> TYPES => TYPES [ I <- integer ] </variable-types>
         <status> DECLARATIONS </status>

    rule [variable-uninit-bool]: <linking-declarations> var I:Id : boolean => .K ... </linking-declarations>
         <variable-values> VALUES => VALUES [ I <- ?_:Bool ] </variable-values>
         <variable-types> TYPES => TYPES [ I <- boolean ] </variable-types>
         <status> DECLARATIONS </status>

    rule [variable-uninit-enum]: <linking-declarations> var I:Id : T:Id => .K ... </linking-declarations>
         <variable-values> VALUES => VALUES [ I <- ?_:Enum ] </variable-values>
         <variable-types> TYPES => TYPES [ I <- T ] </variable-types>
         <status> DECLARATIONS </status> 

    rule [variable-init-unwrap]: <linking-declarations> var I:Id : T:TypeName = E:Expr => #active( var I : T ) ... </linking-declarations>
         <k> .K => E </k>
         <status> DECLARATIONS </status>

    rule [variable-init-store]: <linking-declarations> #active( var I:Id : T:TypeName ) => .K ... </linking-declarations>
         <k> V:Value => .K </k>
         <variable-values> VALUES => VALUES [ I <- V ] </variable-values>
         <variable-types> TYPES => TYPES [ I <- T ] </variable-types>
         <status> DECLARATIONS </status>

    //cleanup
    rule [declarations-cleanup-and-next]: <linking-declarations> .Declarations => .K ... </linking-declarations>
         <status> DECLARATIONS => INIT </status>
endmodule

module TRAN-INIT
    imports CONFIGURATION
    imports XSTS-SYNTAX

    //passing to evaluator
    rule [init-to-evaluator]: <init> E:Sequence => .K </init>
         <evaluator> .K => E </evaluator>
         <status> INIT => EVAL </status>
         <return-status> NO-RETURN => INIT </return-status> [priority(100)]

    rule [init-to-tran]: <init> .K </init>
         <status> INIT => TRAN </status> [priority(100)]
    
endmodule

module TRAN-TRAN
    imports CONFIGURATION
    imports XSTS-SYNTAX

    //passing to evaluator
    rule [tran-to-evaluator]: <transition> E:Sequence </transition>
         <evaluator> .K => E </evaluator>
         <status> TRAN => EVAL </status>
         <return-status> NO-RETURN => TRAN </return-status> [priority(100)]
         /* by passing in 4 as the return status, it will go to ENV next. This is not pretty, since this way it doesn't allow for any
         * closing operations after the running is done, but before we move on to env.
         * If closing operations are needed, it could possibly be done by <transition> E:... => active(E) </transition>
         * and a new rule <transition> active(E:...) => E </transition> <status> 3 => 4 </status> 
         * which would prevent looping
         */

    /**/
    rule [tran-to-env]: <init> .K </init>
        <status> TRAN => ENV </status> [priority(100)]
    
endmodule

module TRAN-ENV
    imports CONFIGURATION
    imports XSTS-SYNTAX

    //passing to evaluator
    rule [env-to-evaluator]: <env> E:Sequence </env>
         <evaluator> .K => E </evaluator>
         <status> ENV => EVAL </status>
         <return-status> NO-RETURN => ENV </return-status> [priority(100)]
         /* by passing in 3 as the return status, it will go to TRAN next. This is not pretty, since this way it doesn't allow for any
         * closing operations after the running is done, but before we move on to tran.
         * If closing operations are needed, it could possibly be done by <transition> E:... => active(E) </transition>
         * and a new rule <transition> active(E:...) => E </transition> <status> 4 => 3 </status> 
         * which would prevent looping
         */

    /**/
    rule [env-to-tran]: <init> .K </init>
        <status> ENV => TRAN </status> [priority(100)]
    
endmodule

module RUN-TRANSITION
    imports CONFIGURATION
    imports XSTS-SYNTAX
    imports BOOL
    imports EXPR-RULES
    imports K-EQUAL

    //prepare operations
    rule [select-next-operation]: <evaluator> O:Operation S:Sequence => O ~> S ... </evaluator>
         <status> EVAL </status> 

    //assignment
    rule [assign-unwrap]: <evaluator> I:Id := E:Expr ; => #active( I ) ... </evaluator>
         <k> .K => E </k>
         <status> EVAL </status>

    rule [assign-update]: <evaluator> #active( X:Id ) => .K ... </evaluator>
         <k> V:Value => .K </k>
         <variable-values> VALUES => VALUES [ X <- V ] </variable-values>
         <status> EVAL </status> //! requires X in_keys( VALUES )
         //! this maybe allows declarations on assign?

    //assume
    rule [assume-unwrap]: <evaluator> assume E:Expr ; => assume #active ; ... </evaluator>
         <k> .K => E </k>
         <status> EVAL </status>
    
    rule [assume-true]: <evaluator> assume #active ; => .K ... </evaluator>
         <k> true => .K </k>
         <status> EVAL </status> //remove assume true;s

    //havoc
    rule [havoc-int]: <evaluator> havoc X:Id ; => .K ... </evaluator>
         <variable-values> VALUES => VALUES [ X <- ?_:Int ] </variable-values>
         <variable-types>... X |-> integer ...</variable-types>
         <status> EVAL </status> //! requires ( X in_keys( VALUES ) ) //andBool ( Id2String( TYPES [I] ) ==String "integer" ) 
         //! this maybe allows declarations on havoc?

    rule [havoc-bool]: <evaluator> havoc I:Id ; => .K ... </evaluator>
         <variable-values> VALUES => VALUES [ I <- ?_:Bool ] </variable-values>
         <variable-types>... I |-> boolean ...</variable-types>
         <status> EVAL </status> //! requires ( I in_keys( VALUES ) ) //andBool ( Id2String( TYPES [I] ) ==String "boolean" ) 
         //! this maybe allows declarations on havoc?

    rule [havoc-enum]: <evaluator> havoc I:Id ; => .K ... </evaluator>
         <variable-values> VALUES => VALUES [ I <- ?_:Enum ] </variable-values>
         <variable-types>... I |-> ENUMTYPE ...</variable-types> 
         <status> EVAL </status> 
         requires /*//! (I in_keys( VALUES )) andBool */ isEnumType( ENUMTYPE ) [owise]
         //! this maybe allows declarations on havoc?

    //choice
    rule [choice-unwrap]: <evaluator> choice O:OrBlocks => O ... </evaluator>
                          <status> EVAL </status>

    //choice: nondet
    rule [choice-first]: <evaluator> { S:Sequence } or _:OrBlocks => S ... </evaluator>
                         <status> EVAL </status>
    rule [choice-rest]: <evaluator> _:Block or O:OrBlocks => O ... </evaluator>
                        <status> EVAL </status> requires O =/=K .OrBlocks

    //cleanup
    rule [evaluator-clean-sequence]: <evaluator> .Sequence => .K ... </evaluator>
         <status> EVAL </status>

    //property verification
    rule [property-start]: <evaluator> .K </evaluator>
         <k> .K => PROP </k>
         <property> PROP:Expr </property>
         <status> EVAL => PROPERTY-EVALUATION-OR-FAIL </status> [priority(200)] 
         //works without priority because there is no <evaluator>.K</evaluator> <status>100</status> state before this one

    //only exit evaluator if property is true
    rule [property-true]: <k> true => .K </k>
         <status> PROPERTY-EVALUATION-OR-FAIL => PROPERTY-SUCCESS </status>
    
    rule [property-false]: <k> false </k>
         <status> PROPERTY-EVALUATION-OR-FAIL => PROPERTY-FAIL </status>

    //exit
    rule [evaluator-exit]: <evaluator> .K </evaluator>
         <return-status> RETURN:Status => NO-RETURN </return-status>
         <status> PROPERTY-SUCCESS => RETURN </status> [priority(1000)]

endmodule

module XSTS
    imports PROGRAM-LINKING
    imports DECLARATIONS
    imports TRAN-INIT

    // Important! This will currently enter an infinite loop, krun only with --depth bound!
    imports TRAN-TRAN
    imports TRAN-ENV

    imports RUN-TRANSITION
endmodule