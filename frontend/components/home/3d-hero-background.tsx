"use client"

import { useRef } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { PerspectiveCamera } from "@react-three/drei"

function FloatingCube({ position, color, speed }) {
  const meshRef = useRef(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += speed
      meshRef.current.rotation.y += speed * 0.7
      meshRef.current.position.y += Math.sin(state.clock.elapsedTime * speed) * 0.003
    }
  })

  return (
    <mesh ref={meshRef} position={position}>
      <boxGeometry args={[1, 1, 1]} />
      <meshPhongMaterial color={color} emissive={color} emissiveIntensity={0.3} wireframe={true} />
    </mesh>
  )
}

function FloatingSphere({ position, color, speed }) {
  const meshRef = useRef(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += speed * 0.5
      meshRef.current.rotation.z += speed * 0.3
      meshRef.current.position.y += Math.cos(state.clock.elapsedTime * speed) * 0.004
    }
  })

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.8, 32, 32]} />
      <meshPhongMaterial color={color} emissive={color} emissiveIntensity={0.5} />
    </mesh>
  )
}

function SecurityShield({ position, speed }) {
  const meshRef = useRef(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z += speed
      meshRef.current.position.x += Math.sin(state.clock.elapsedTime * speed * 0.5) * 0.002
    }
  })

  return (
    <mesh ref={meshRef} position={position}>
      <octahedronGeometry args={[0.7]} />
      <meshPhongMaterial color="#a855f7" emissive="#d946ef" emissiveIntensity={0.6} wireframe={false} />
    </mesh>
  )
}

function Scene() {
  return (
    <>
      <PerspectiveCamera position={[0, 0, 20]} fov={50} makeDefault />
      <ambientLight intensity={0.5} color="#ffffff" />
      <pointLight position={[10, 10, 10]} intensity={1.5} color="#a855f7" />
      <pointLight position={[-10, -10, 10]} intensity={1} color="#fbbf24" />

      {/* Central cube */}
      <FloatingCube position={[0, 0, 0]} color="#a855f7" speed={0.008} />

      {/* Surrounding security shields */}
      <SecurityShield position={[5, 5, -5]} speed={0.005} />
      <SecurityShield position={[-5, 5, -5]} speed={0.006} />
      <SecurityShield position={[5, -5, -5]} speed={0.004} />
      <SecurityShield position={[-5, -5, -5]} speed={0.007} />

      {/* Data cubes */}
      <FloatingCube position={[8, 0, -10]} color="#06b6d4" speed={0.005} />
      <FloatingCube position={[-8, 0, -10]} color="#f59e0b" speed={0.006} />
      <FloatingCube position={[0, 8, -10]} color="#ec4899" speed={0.004} />
      <FloatingCube position={[0, -8, -10]} color="#06b6d4" speed={0.007} />

      {/* Floating analysis spheres */}
      <FloatingSphere position={[3, 3, -8]} color="#a855f7" speed={0.004} />
      <FloatingSphere position={[-3, -3, -8]} color="#fbbf24" speed={0.005} />
      <FloatingSphere position={[3, -3, -8]} color="#06b6d4" speed={0.006} />
      <FloatingSphere position={[-3, 3, -8]} color="#ec4899" speed={0.004} />
    </>
  )
}

export function HeroBackground() {
  return (
    <div className="absolute inset-0 -z-10">
      <Canvas
        style={{
          width: "100%",
          height: "100%",
        }}
        performance={{ min: 0.5 }}
      >
        <color attach="background" args={["#0a0e27"]} />
        <Scene />
      </Canvas>
    </div>
  )
}
