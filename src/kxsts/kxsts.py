from __future__ import annotations

from pyk.kcfg.show import NodePrinter

__all__ = ['KXsts']

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, final

from pyk.cli.utils import check_dir_path
from pyk.cterm.symbolic import CTermSymbolic
from pyk.kast.formatter import Formatter
from pyk.kast.inner import KSequence, KVariable
from pyk.kast.manip import ml_pred_to_bool
from pyk.kast.outer import read_kast_definition
from pyk.kcfg.explore import KCFGExplore
from pyk.kcfg.semantics import DefaultSemantics
from pyk.kore.rpc import KoreServer, KoreClient
from pyk.ktool.claim_loader import ClaimLoader
from pyk.ktool.kprove import KProve
from pyk.proof.reachability import APRProof, APRProver
from pyk.proof.show import APRProofShow
from pyk.proof.tui import APRProofViewer
from pyk.utils import single

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from typing import Final

    from pyk.cterm.cterm import CTerm
    from pyk.kast.outer import KDefinition
    from pyk.kcfg.kcfg import KCFG
    from pyk.kore.syntax import Pattern

_LOGGER: Final = logging.getLogger(__name__)


class XstsSemantics(DefaultSemantics):
    def is_terminal(self, c: CTerm) -> bool:
        # lefutott a property kiértékelés, és false lett az eredmény
        k_cell = c.cell('K_CELL')
        if type(k_cell) is KSequence:
            if len(k_cell) == 0:
                return True
            if len(k_cell) == 1 and type(k_cell[0]) is KVariable:
                return True
        if type(k_cell) is KVariable:
            return True
        return False
    
    # pretend we don't have loops

@final
@dataclass(frozen=True)
class XstsDist:
    source_dir: Path
    # llvm_dir: Path
    # llvm_lib_dir: Path
    haskell_dir: Path

    def __init__(
        self,
        *,
        source_dir: str | Path,
        # llvm_dir: str | Path,
        # llvm_lib_dir: str | Path,
        haskell_dir: str | Path,
    ):
        source_dir = Path(source_dir)
        check_dir_path(source_dir)

        # llvm_dir = Path(llvm_dir)
        # check_dir_path(llvm_dir)

        # llvm_lib_dir = Path(llvm_lib_dir)
        # check_dir_path(llvm_lib_dir)

        haskell_dir = Path(haskell_dir)
        check_dir_path(haskell_dir)

        object.__setattr__(self, 'source_dir', source_dir)
        # object.__setattr__(self, 'llvm_dir', llvm_dir)
        # object.__setattr__(self, 'llvm_lib_dir', llvm_lib_dir)
        object.__setattr__(self, 'haskell_dir', haskell_dir)

    @staticmethod
    def load() -> XstsDist:
        return XstsDist(
            source_dir=XstsDist._find('source'),
            # llvm_dir=XstsDist._find('llvm'),
            # llvm_lib_dir=XstsDist._find('llvm-lib'),
            haskell_dir=XstsDist._find('haskell'),
        )

    @staticmethod
    def _find(target: str) -> Path:
        """
        Find a `kdist` target:
        * if KXSTS_<TARGET_IN_SCREAMING_SNAKE_CASE>_DIR is set --- use that
        * otherwise ask `kdist`
        """

        from os import getenv

        from pyk.kdist import kdist

        env_dir = getenv(f"KXSTS_{target.replace('-', '_').upper()}_DIR")
        if env_dir:
            path = Path(env_dir)
            check_dir_path(path)
            _LOGGER.info(f'Using target at {path}')
            return path

        return kdist.get(f'kxsts-semantics.{target}')


@final
@dataclass(frozen=True)
class KXsts:
    dist: XstsDist
    proof_dir: Path

    def __init__(self) -> None:
        dist = XstsDist.load()

        proof_dir = Path('.') / '.kxsts' / 'proofs'
        proof_dir.mkdir(exist_ok=True, parents=True)

        object.__setattr__(self, 'dist', dist)
        object.__setattr__(self, 'proof_dir', proof_dir)

    @cached_property
    def definition(self) -> KDefinition:
        return read_kast_definition(self.dist.haskell_dir / 'compiled.json') # does this work with haskell?
    
    @cached_property
    def format(self) -> Formatter:
        return Formatter(self.definition)
    
    @cached_property
    def kprove(self) -> KProve:
        return KProve(definition_dir=self.dist.haskell_dir, use_directory=self.proof_dir)
    
    #! Is this something I'm using?
    def run(
        self,
        pattern: Pattern,
        *,
        depth: int | None = None,
    ) -> Pattern:
        from pyk.ktool.krun import llvm_interpret, _krun
        # from pyk.ktool.krun import haskell_interpret

        return _krun(definition_dir=self.dir.haskell_dir, pattern=pattern, depth=depth)
        # return llvm_interpret(definition_dir=self.dist.haskell_dir, pattern=pattern, depth=depth)

    #! Is this something I'm using?
    def pattern(self, *, pgm: str, env: Mapping[str, int]) -> Pattern:
        from pyk.kore.prelude import BOOL, ID, INT, SORT_K_ITEM, bool_dv, inj, int_dv, map_pattern, top_cell_initializer
        from pyk.kore.syntax import DV, SortApp, String

        def inj_dv(val: int) -> Pattern:
            if isinstance(val, bool):
                return inj(BOOL, SORT_K_ITEM, bool_dv(val))
            return inj(INT, SORT_K_ITEM, int_dv(val))

        pgm_pattern = self.parse(pgm)
        env_pattern = map_pattern(
            *(
                (
                    inj(ID, SORT_K_ITEM, DV(ID, String(var))),
                    inj_dv(val),
                )
                for var, val in env.items()
            )
        )
        return top_cell_initializer(
            {
                '$PGM': inj(SortApp('SortStmt'), SORT_K_ITEM, pgm_pattern),
                '$ENV': inj(SortApp('SortMap'), SORT_K_ITEM, env_pattern),
            }
        )

    #! Is this something I'm using?
    def parse(self, pgm: str) -> Pattern:
        from subprocess import CalledProcessError

        from pyk.kore.parser import KoreParser
        from pyk.utils import run_process_2

        parser = self.dist.haskell_dir / 'parser_PGM' # does this work with haskell?
        args = [str(parser), '/dev/stdin']

        try:
            kore_text = run_process_2(args, input=pgm).stdout
        except CalledProcessError as err:
            raise ValueError(err.stderr) from err

        return KoreParser(kore_text).pattern()

    #! Is this something I'm using?
    def pretty(self, pattern: Pattern, color: bool | None = None) -> str:
        from pyk.kore.tools import kore_print

        return kore_print(pattern, definition_dir=self.dist.haskell_dir, color=bool(color)) # does this work with haskell?
    
    #! Is this something I'm using?
    def env(self, pattern: Pattern) -> dict[str, int]:
        import pyk.kore.match as km
        from pyk.kore.prelude import BOOL, INT
        from pyk.utils import case, chain

        extract = (
            chain
            >> km.app("Lbl'-LT-'generatedTop'-GT-'")
            >> km.arg("Lbl'-LT-'env'-GT-'")
            >> km.arg(0)
            >> km.kore_map_of(
                key=chain >> km.inj >> km.kore_id,
                value=chain
                >> km.match_inj
                >> case(
                    (
                        (
                            lambda inj: inj.sorts[0] == BOOL,
                            chain >> km.arg(0) >> km.kore_bool,
                        ),
                        (
                            lambda inj: inj.sorts[0] == INT,
                            chain >> km.arg(0) >> km.kore_int,
                        ),
                    )
                ),
            )
        )

        try:
            return dict(extract(pattern))
        except Exception as err:
            pretty_pattern = self.pretty(pattern)
            raise ValueError(f'Cannot extract environment from pattern:\n{pretty_pattern}') from err
        
    #! Is this something I'm using?
    def debug(self, pattern: Pattern) -> Callable[[int | None], None]:
        """Return a closure that enables step-by-step debugging in a REPL.

        Each call to the function pretty-prints the resulting configuration.

        Example:
            step = KImp().debug(pattern)
            step()            # Run a single step
            step(1)           # Run a single step
            step(0)           # Just print the current configuration
            step(depth=None)  # Run to completion
        """

        def step(depth: int | None = 1) -> None:
            nonlocal pattern
            pattern = self.run(pattern, depth=depth)
            print(self.pretty(pattern, color=True))

        return step
    
    def prove(
            self,
            *,
            spec_file: str,
            spec_module: str,
            claim_id: str,
            max_iterations: int,
            max_depth: int,
            reinit: bool,
            includes: Iterable[str | Path] | None = None,
        ) -> None:
            include_dirs = [self.dist.source_dir / 'kxsts-semantics'] + (
                [Path(include) for include in includes] if includes is not None else []
            )

            claims = ClaimLoader(self.kprove).load_claims(
                Path(spec_file),
                spec_module_name=spec_module,
                claim_labels=[claim_id],
                include_dirs=include_dirs,
            )
            claim = single(claims)
            spec_label = f'{spec_module}.{claim_id}'

            if not reinit and APRProof.proof_data_exists(spec_label, self.proof_dir):
                # load an existing proof (to continue work on it)
                proof = APRProof.read_proof_data(proof_dir=self.proof_dir, id=f'{spec_module}.{claim_id}')
            else:
                # ignore existing proof data and reinitilize it from a claim
                proof = APRProof.from_claim(self.kprove.definition, claim=claim, logs={}, proof_dir=self.proof_dir)

            with self.explore(id=spec_label) as kcfg_explore:
                prover = APRProver(
                    kcfg_explore=kcfg_explore,
                    execute_depth=max_depth,
                    cut_point_rules=['XSTS.property-true'], # the model is 'stable' at this point
                    terminal_rules=['XSTS.property-false'], # something has violated the state invariant property
                )
                prover.advance_proof(proof, max_iterations=max_iterations)

                print(proof.summary)
                print('============================================')
                print("What's next?: ")
                print('============================================')
                print('To inspect the symbolic execution trace interactively, run: ')
                print(f'  kxsts view {spec_module} {claim_id}')
                print('============================================')
                print('To dump the symbolic execution trace into stdout, run: ')
                print(f'  kxsts show {spec_module} {claim_id}')
                print('============================================')
                if not proof.passed:
                    print('To retry the failed/pending proof, run : ')
                    print(f'  kxsts prove {spec_file} {spec_module} {claim_id}')
                print('To start the proof from scratch: ')
                print(f'  kxsts prove --reinit {spec_file} {spec_module} {claim_id}')

    @contextmanager
    def explore(self, *, id: str) -> Iterator[KCFGExplore]:
        with KoreServer(#BoosterServer( # when using KoreServer comment out llvm lines
            {
                'kompiled_dir': self.kprove.definition_dir,
                # 'llvm_kompiled_dir': self.dist.llvm_lib_dir, # this might not work without llvm
                'module_name': self.kprove.main_module,
            }
        ) as server:
            with KoreClient('localhost', server.port) as client:
                cterm_symbolic = cterm_symbolic = CTermSymbolic(
                    kore_client=client,
                    definition=self.kprove.definition,
                )
                yield KCFGExplore(
                    kcfg_semantics=XstsSemantics(),
                    id=id,
                    cterm_symbolic=cterm_symbolic,
                )
    
    #! Is this something I'm using?
    def view_kcfg(
        self,
        spec_module: str,
        claim_id: str,
    ) -> None:
        proof = APRProof.read_proof_data(proof_dir=self.proof_dir, id=f'{spec_module}.{claim_id}')
        kcfg_viewer = APRProofViewer(proof, self.kprove, node_printer=XstsNodePrinter(kxsts=self))
        kcfg_viewer.run()

    #! Is this something I'm using?
    def show_kcfg(
        self,
        spec_module: str,
        claim_id: str,
    ) -> None:
        proof = APRProof.read_proof_data(proof_dir=self.proof_dir, id=f'{spec_module}.{claim_id}')
        proof_show = APRProofShow(self.definition, node_printer=XstsNodePrinter(kxsts=self))
        res_lines = proof_show.show(
            proof,
        )
        print('\n'.join(res_lines))


class XstsNodePrinter(NodePrinter):
    kxsts: KXsts

    def __init__(self, kxsts: KXsts):
        from pyk.cterm.show import CTermShow

        super().__init__(cterm_show=CTermShow(kxsts.format))
        self.kxsts = kxsts

    def print_node(self, kcfg: KCFG, node: KCFG.Node) -> list[str]:
        res = super().print_node(kcfg, node)

        k_cell = node.cterm.cell('K_CELL')
        env_cell = node.cterm.cell('ENV_CELL')

        # pretty-print the configuration
        res += self.kxsts.format(k_cell).splitlines()
        res += ['env:']
        res += [f'  {line}' for line in self.kxsts.format(env_cell).splitlines()]

        # pretty-print the constraints
        constraints = [ml_pred_to_bool(c) for c in node.cterm.constraints]
        if len(constraints) > 0:
            res += ['constraints:']
            res += [f'  {self.kxsts.format(c)}' for c in constraints]

        return res