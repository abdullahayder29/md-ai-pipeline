#!/usr/bin/env nextflow

/*
================================================================================
  MD-AI PIPELINE: Nextflow-Orchestrated Molecular Dynamics + AI Analysis
================================================================================
  Author      : Bioinformatics Pipeline Template
  Description : A modular, reproducible pipeline that runs GROMACS MD simulations
                and feeds trajectory data into a PyTorch Autoencoder for
                conformational space analysis.

  HOW NEXTFLOW WORKS (Beginner Note):
  ------------------------------------
  Nextflow uses a "dataflow" programming model. Instead of writing a sequential
  script, you define PROCESSES (units of work) and CHANNELS (data streams that
  connect processes). Nextflow automatically figures out the correct execution
  order based on which channels feed into which processes. This is the key to
  its power: parallelism and dependency management are handled FOR you.

  PIPELINE STAGES:
    1. PREPARE_SYSTEM   → Build GROMACS topology and run energy minimization
    2. PRODUCTION_MD    → Run the full production simulation
    3. PROCESS_TRAJECTORY → Fix periodic boundary conditions (PBC)
    4. AI_ANALYSIS      → PyTorch Autoencoder dimensionality reduction
================================================================================
*/

// ─────────────────────────────────────────────────────────────────────────────
// NEXTFLOW DSL2 DECLARATION
// DSL2 allows you to define processes in separate files and reuse them.
// It is the modern, recommended way to write Nextflow pipelines.
// ─────────────────────────────────────────────────────────────────────────────
nextflow.enable.dsl=2

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE PARAMETERS
// 'params' is a special Nextflow object. Parameters defined here can be
// overridden at runtime from the command line:
//   nextflow run main.nf --pdb_file my_protein.pdb --sim_time 100
// This makes your pipeline flexible without touching the source code.
// ─────────────────────────────────────────────────────────────────────────────
params {
    // --- Input Files ---
    // Path to the input protein structure (PDB format)
    pdb_file        = "${projectDir}/data/input/protein.pdb"

    // Path to GROMACS forcefield and MD parameter files
    forcefield      = "amber99sb-ildn"          // Force field to use
    water_model     = "tip3p"                   // Water model
    mdp_em          = "${projectDir}/data/mdp/em.mdp"           // Energy minimization params
    mdp_nvt         = "${projectDir}/data/mdp/nvt.mdp"          // NVT equilibration params
    mdp_npt         = "${projectDir}/data/mdp/npt.mdp"          // NPT equilibration params
    mdp_md          = "${projectDir}/data/mdp/md.mdp"           // Production MD params

    // --- Simulation Settings ---
    sim_time        = 10        // Simulation time in nanoseconds (ns)
    box_size        = 1.0       // Minimum distance from protein to box edge (nm)

    // --- AI Analysis Settings ---
    latent_dim      = 2         // Autoencoder bottleneck dimensions (for 2D visualization)
    n_epochs        = 150       // Number of training epochs for the autoencoder
    batch_size      = 64        // Mini-batch size for PyTorch DataLoader

    // --- Output Directory ---
    // All results will be published here
    outdir          = "${projectDir}/results"
}

// ─────────────────────────────────────────────────────────────────────────────
// LOG BANNER
// A professional pipeline always prints a summary at startup so the user
// knows exactly what settings are being used.
// ─────────────────────────────────────────────────────────────────────────────
log.info """
╔══════════════════════════════════════════════════════════════╗
║           MD-AI CONFORMATIONAL ANALYSIS PIPELINE             ║
╚══════════════════════════════════════════════════════════════╝
  Input PDB         : ${params.pdb_file}
  Force Field        : ${params.forcefield}
  Simulation Time    : ${params.sim_time} ns
  Latent Dimensions  : ${params.latent_dim}
  AE Training Epochs : ${params.n_epochs}
  Output Directory   : ${params.outdir}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".stripIndent()


// ─────────────────────────────────────────────────────────────────────────────
// PROCESS 1: PREPARE_SYSTEM
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: Take a raw PDB file and set up a complete GROMACS simulation system.
// This is analogous to "lab prep" — before running an experiment, you must
// set up your equipment (topology, solvation, ion addition, energy minimization).
//
// NEXTFLOW CONCEPT — `publishDir`:
//   The `publishDir` directive tells Nextflow to COPY the output files from the
//   temporary work directory to a permanent results folder. Without this, outputs
//   are buried in Nextflow's internal 'work/' directory.
//
// NEXTFLOW CONCEPT — `input` / `output` blocks:
//   These define the "interface" of the process. Nextflow uses these to wire
//   processes together automatically via channels.
// ─────────────────────────────────────────────────────────────────────────────
process PREPARE_SYSTEM {
    tag "System Setup: ${pdb.simpleName}"    // A label shown in logs for this job

    publishDir "${params.outdir}/01_system_prep", mode: 'copy'

    // Declare which software this process needs (managed by conda/modules/docker)
    conda "bioconda::gromacs=2023.3"

    input:
    path pdb             // The input PDB file, passed in as a channel item
    path mdp_em          // Energy minimization parameter file

    output:
    path "em.gro",       emit: minimized_structure   // Named output channel
    path "topol.top",    emit: topology
    path "*.itp",        emit: itp_files
    path "em.log",       emit: em_log

    // The `script` block contains the actual shell commands to run.
    // Variables like ${pdb} are automatically substituted by Nextflow.
    script:
    """
    # ── STEP 1: Generate GROMACS topology from PDB ──────────────────────────
    # 'pdb2gmx' reads the PDB, applies the force field, and creates:
    #   - topol.top : the full system topology (bonds, angles, charges)
    #   - processed.gro : the GROMACS coordinate file
    echo "${params.forcefield}" | gmx pdb2gmx \\
        -f ${pdb} \\
        -o processed.gro \\
        -water ${params.water_model} \\
        -ignh                         # Ignore existing hydrogens; let GROMACS add them

    # ── STEP 2: Define the simulation box ───────────────────────────────────
    # 'editconf' creates a periodic boundary box around the protein.
    # '-bt dodecahedron' is efficient for globular proteins (saves water molecules).
    gmx editconf \\
        -f processed.gro \\
        -o boxed.gro \\
        -c \\                          # Center the protein in the box
        -d ${params.box_size} \\      # Minimum protein-to-wall distance (nm)
        -bt dodecahedron

    # ── STEP 3: Solvate the system ─────────────────────────────────────────
    # 'solvate' fills the box with water molecules.
    gmx solvate \\
        -cp boxed.gro \\
        -cs spc216.gro \\             # Pre-built water box (comes with GROMACS)
        -o solvated.gro \\
        -p topol.top                  # Automatically updates the topology file

    # ── STEP 4: Add ions to neutralize the system ───────────────────────────
    # Biological systems need counter-ions (Na+/Cl-) for charge neutrality.
    gmx grompp \\
        -f ${mdp_em} \\
        -c solvated.gro \\
        -p topol.top \\
        -o ions.tpr \\
        -maxwarn 1

    echo "SOL" | gmx genion \\
        -s ions.tpr \\
        -o ionized.gro \\
        -p topol.top \\
        -pname NA \\
        -nname CL \\
        -neutral                      # Add just enough ions to neutralize

    # ── STEP 5: Energy Minimization ─────────────────────────────────────────
    # Before running dynamics, relax the system to remove bad contacts.
    # 'grompp' preprocesses the topology + parameters → binary .tpr file
    gmx grompp \\
        -f ${mdp_em} \\
        -c ionized.gro \\
        -p topol.top \\
        -o em.tpr

    # 'mdrun' is the core simulation engine; '-v' makes it verbose
    gmx mdrun -v -deffnm em
    """
}


// ─────────────────────────────────────────────────────────────────────────────
// PROCESS 2: PRODUCTION_MD
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: Run the full production molecular dynamics simulation.
// This is the computationally intensive step that generates the trajectory.
//
// NEXTFLOW CONCEPT — `label`:
//   Labels map processes to resource profiles defined in nextflow.config.
//   By using `label 'gpu'`, we tell Nextflow that this process should run
//   on a GPU node with the resources specified in the config file.
//   This keeps resource allocation SEPARATE from the scientific logic.
// ─────────────────────────────────────────────────────────────────────────────
process PRODUCTION_MD {
    tag "Production MD"
    label 'gpu'            // This process requires GPU resources (see nextflow.config)

    publishDir "${params.outdir}/02_production_md", mode: 'copy'

    conda "bioconda::gromacs=2023.3"

    input:
    path minimized_gro    // Output from PREPARE_SYSTEM
    path topology         // topol.top from PREPARE_SYSTEM
    path itp_files        // Supporting topology files
    path mdp_nvt          // NVT equilibration parameters
    path mdp_npt          // NPT equilibration parameters
    path mdp_md           // Production MD parameters

    output:
    path "md.xtc",        emit: trajectory      // The compressed trajectory file
    path "md.gro",        emit: final_structure // Final coordinates
    path "md.tpr",        emit: tpr_file        // Binary run input (needed for analysis)
    path "md.log",        emit: md_log

    script:
    """
    # ── STEP 1: NVT Equilibration ────────────────────────────────────────────
    # NVT = constant Number of particles, Volume, and Temperature.
    # We "heat" the system gradually to the target temperature (e.g., 300 K).
    gmx grompp \\
        -f ${mdp_nvt} \\
        -c ${minimized_gro} \\
        -p ${topology} \\
        -o nvt.tpr \\
        -maxwarn 1

    gmx mdrun -v -deffnm nvt -ntmpi 1 -ntomp ${task.cpus} -gpu_id 0

    # ── STEP 2: NPT Equilibration ─────────────────────────────────────────────
    # NPT = constant Number of particles, Pressure, and Temperature.
    # We now equilibrate the PRESSURE (density) of the system.
    gmx grompp \\
        -f ${mdp_npt} \\
        -c nvt.gro \\
        -t nvt.cpt \\                 # Continue from NVT checkpoint
        -p ${topology} \\
        -o npt.tpr \\
        -maxwarn 1

    gmx mdrun -v -deffnm npt -ntmpi 1 -ntomp ${task.cpus} -gpu_id 0

    # ── STEP 3: Production MD ─────────────────────────────────────────────────
    # The real simulation! This generates the trajectory we will analyze.
    gmx grompp \\
        -f ${mdp_md} \\
        -c npt.gro \\
        -t npt.cpt \\
        -p ${topology} \\
        -o md.tpr

    gmx mdrun -v -deffnm md \\
        -ntmpi 1 \\
        -ntomp ${task.cpus} \\
        -gpu_id 0 \\
        -nb gpu \\                    # Non-bonded interactions on GPU
        -pme gpu                      # Particle Mesh Ewald electrostatics on GPU
    """
}


// ─────────────────────────────────────────────────────────────────────────────
// PROCESS 3: PROCESS_TRAJECTORY
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: Clean the raw trajectory for analysis.
// MD simulations use Periodic Boundary Conditions (PBC): when an atom leaves
// one side of the box, it re-enters from the other side. This causes visual
// "jumps" in the trajectory. This process fixes those artifacts.
// ─────────────────────────────────────────────────────────────────────────────
process PROCESS_TRAJECTORY {
    tag "Trajectory Processing"

    publishDir "${params.outdir}/03_processed_trajectory", mode: 'copy'

    conda "bioconda::gromacs=2023.3"

    input:
    path trajectory    // Raw md.xtc trajectory
    path tpr_file      // md.tpr — contains system topology needed for PBC fix
    path final_gro     // Final structure for reference

    output:
    path "protein_ca.xtc",  emit: clean_trajectory    // Clean, analysis-ready trajectory
    path "protein_ca.pdb",  emit: reference_structure  // Reference structure for MDAnalysis

    script:
    """
    # ── STEP 1: Fix Periodic Boundary Conditions ─────────────────────────────
    # 'trjconv' is GROMACS's Swiss Army knife for trajectory manipulation.
    # '-pbc mol' ensures molecules are made whole (not split across box boundaries).
    # '-center' centers the protein in the box.
    # We use echo to auto-answer the interactive prompts: "1 0" means:
    #   - Group 1 (Protein) for centering
    #   - Group 0 (System) for output
    echo "1 0" | gmx trjconv \\
        -s ${tpr_file} \\
        -f ${trajectory} \\
        -o centered.xtc \\
        -center \\
        -pbc mol \\
        -ur compact

    # ── STEP 2: Extract only Cα atoms ────────────────────────────────────────
    # For conformational analysis, we typically use only backbone Cα atoms.
    # This reduces dimensionality significantly (e.g., 10,000 atoms → 300 Cα atoms)
    # without losing the overall protein shape information.
    # "3 3" = select group 3 (C-alpha) for both fitting and output
    echo "3 3" | gmx trjconv \\
        -s ${tpr_file} \\
        -f centered.xtc \\
        -o protein_ca.xtc \\
        -fit rot+trans \\            # Align frames to remove rotation/translation
        -n

    # ── STEP 3: Extract reference structure ──────────────────────────────────
    # Save first frame as PDB reference (used by the Python analysis script)
    echo "3" | gmx trjconv \\
        -s ${tpr_file} \\
        -f protein_ca.xtc \\
        -o protein_ca.pdb \\
        -dump 0                      # Dump only time=0 (first frame)
    """
}


// ─────────────────────────────────────────────────────────────────────────────
// PROCESS 4: AI_ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────
// PURPOSE: Run the PyTorch Autoencoder to perform non-linear dimensionality
// reduction on the MD trajectory.
//
// NEXTFLOW CONCEPT — The `bin/` directory:
//   Any executable script placed in the `bin/` directory is automatically
//   added to the PATH for all processes. This is the standard Nextflow pattern
//   for separating pipeline scripts from workflow logic. We call our Python
//   script as if it were a system command.
//
// NEXTFLOW CONCEPT — Separating concerns:
//   The Nextflow process handles ORCHESTRATION (inputs, outputs, resources).
//   The Python script handles the SCIENCE (machine learning, visualization).
//   This separation makes both easier to maintain and test independently.
// ─────────────────────────────────────────────────────────────────────────────
process AI_ANALYSIS {
    tag "PyTorch Autoencoder"

    publishDir "${params.outdir}/04_ai_analysis", mode: 'copy'

    // Conda environment with all Python/ML dependencies
    conda "pytorch::pytorch=2.2.0 conda-forge::mdanalysis=2.7.0 conda-forge::matplotlib=3.8.0 conda-forge::seaborn=0.13.0 conda-forge::scikit-learn=1.4.0"

    input:
    path clean_trajectory     // Cα trajectory from PROCESS_TRAJECTORY
    path reference_structure  // PDB reference structure

    output:
    path "latent_space.png",          emit: plot_latent       // 2D latent space visualization
    path "reconstruction_loss.png",   emit: plot_loss         // Training loss curve
    path "latent_coordinates.csv",    emit: latent_coords     // Raw data for downstream use
    path "autoencoder_model.pt",      emit: model_weights     // Saved PyTorch model

    script:
    // We call our Python script from the bin/ directory with all parameters
    // passed as command-line arguments. This keeps the script reusable.
    """
    conformational_analysis.py \\
        --trajectory    ${clean_trajectory} \\
        --topology      ${reference_structure} \\
        --latent-dim    ${params.latent_dim} \\
        --epochs        ${params.n_epochs} \\
        --batch-size    ${params.batch_size} \\
        --output-prefix .
    """
}


// ─────────────────────────────────────────────────────────────────────────────
// WORKFLOW BLOCK — The Pipeline Orchestrator
// ─────────────────────────────────────────────────────────────────────────────
// This is the heart of the pipeline. The `workflow` block wires the processes
// together using CHANNELS. Think of channels as conveyor belts: each process
// produces outputs that get placed on a channel, which delivers them to the
// next process as inputs.
//
// Nextflow resolves the execution order AUTOMATICALLY based on these data
// dependencies. If PRODUCTION_MD needs output from PREPARE_SYSTEM, Nextflow
// guarantees PREPARE_SYSTEM runs first — you never need to manage this yourself.
// ─────────────────────────────────────────────────────────────────────────────
workflow {

    // ── Create input channels from parameters ──────────────────────────────
    // `Channel.fromPath()` creates a channel that emits file paths.
    // These are the "source" channels — the pipeline's entry points.
    pdb_ch    = Channel.fromPath(params.pdb_file,  checkIfExists: true)
    mdp_em_ch = Channel.fromPath(params.mdp_em,    checkIfExists: true)
    mdp_nvt_ch = Channel.fromPath(params.mdp_nvt,  checkIfExists: true)
    mdp_npt_ch = Channel.fromPath(params.mdp_npt,  checkIfExists: true)
    mdp_md_ch  = Channel.fromPath(params.mdp_md,   checkIfExists: true)

    // ── Stage 1: System Preparation ────────────────────────────────────────
    // Pass the PDB file and EM parameters into the first process.
    // The `.out` object holds all the named output channels we declared.
    PREPARE_SYSTEM(pdb_ch, mdp_em_ch)

    // ── Stage 2: Production MD ─────────────────────────────────────────────
    // We collect the outputs from PREPARE_SYSTEM and feed them into PRODUCTION_MD.
    // `.collect()` gathers all items in a channel into a single list — useful
    // when a process emits multiple files (like the *.itp files).
    PRODUCTION_MD(
        PREPARE_SYSTEM.out.minimized_structure,
        PREPARE_SYSTEM.out.topology,
        PREPARE_SYSTEM.out.itp_files.collect(),
        mdp_nvt_ch,
        mdp_npt_ch,
        mdp_md_ch
    )

    // ── Stage 3: Trajectory Processing ─────────────────────────────────────
    PROCESS_TRAJECTORY(
        PRODUCTION_MD.out.trajectory,
        PRODUCTION_MD.out.tpr_file,
        PRODUCTION_MD.out.final_structure
    )

    // ── Stage 4: AI Analysis ─────────────────────────────────────────────
    AI_ANALYSIS(
        PROCESS_TRAJECTORY.out.clean_trajectory,
        PROCESS_TRAJECTORY.out.reference_structure
    )

    // ── Completion Summary ─────────────────────────────────────────────────
    // `.view()` subscribes to a channel and prints each item — great for logging.
    AI_ANALYSIS.out.plot_latent.view { f ->
        log.info "✅ Pipeline complete! Latent space visualization: ${f}"
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// WORKFLOW COMPLETION HANDLER
// This block runs after the entire workflow finishes — success or failure.
// It's used for cleanup, notifications, or summary reporting.
// ─────────────────────────────────────────────────────────────────────────────
workflow.onComplete {
    def status = workflow.success ? "✅ SUCCESS" : "❌ FAILED"
    log.info """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Pipeline Status  : ${status}
    Duration         : ${workflow.duration}
    Results Dir      : ${params.outdir}
    Work Dir         : ${workflow.workDir}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """.stripIndent()
}
